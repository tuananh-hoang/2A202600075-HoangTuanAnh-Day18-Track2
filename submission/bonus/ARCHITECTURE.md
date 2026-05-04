# Architecture: Multimodal RAG Lakehouse

**Bonus Challenge — Day 18 Track 2 | VinUni AI20k**  
**Problem:** Multimodal RAG — image + text + audio embeddings cho retrieval  
**Author:** 2A202600075-HoangTuanAnh
**Date:** 04/05/2026

---

## 1. Problem Statement

### 1.1 Pain Point

Một AI startup VN xây hệ thống RAG cho internal knowledge base bao gồm tài liệu PDF (policy, spec), hình ảnh quy trình (PNG/JPG), và audio transcript từ meeting. Team kỹ thuật không trả lời được các câu hỏi sau:

- "Embedding nào đang stale — document đã được update nhưng vector chưa được re-generate?"
- "Chunk nào được retrieve nhiều nhất nhưng có relevance score thấp, tức là retrieval đang fail?"
- "Chi phí embedding OpenAI text-embed-3-large tuần này là bao nhiêu, breakdown theo modality?"
- "Khi tôi update một document, lineage từ raw file → chunk → embedding → retrieval log thay đổi như thế nào?"

Hiện tại: embedding vectors nằm rải rác trong vector DB, không có lineage ngược về document gốc, không có freshness tracking, không có cost aggregation.

### 1.2 Success Criteria

| Tiêu chí | Ngưỡng chấp nhận |
|---|---|
| Query embedding freshness status | < 2s |
| Detect stale embeddings (doc_hash mismatch) | Tự động hourly |
| Retrieval quality report: precision@5, MRR | Có hàng ngày |
| Time travel: xem embedding trước khi re-index | `VERSION AS OF N` hoạt động |
| Cost breakdown theo modality + model | Query < 3s |
| Compliance: xóa data theo yêu cầu (Decree-13) | `DELETE` + `VACUUM` hoàn chỉnh |

### 1.3 Scope (1 tuần)

- **3 modalities:** text (PDF), image (PNG — dùng CLIP caption), audio (transcript từ Whisper)
- **Volume:** ~50,000 chunks, ~2,000 documents, ~500k retrieval events/ngày
- **1 embedding provider:** OpenAI `text-embed-3-large` (1536 dims)
- **1 region:** Singapore (ap-southeast-1)
- **Multi-tenant:** 3 customer accounts

---

## 2. Decision Log

### 2.1 Table Format: Delta Lake vs Apache Iceberg vs Apache Hudi

| Tiêu chí | Delta Lake | Apache Iceberg | Apache Hudi |
|---|---|---|---|
| Workload chính (append + MERGE re-embed) | ✅ Tốt | ✅ Tốt | ✅ Upsert-heavy tốt nhất |
| Engine hỗ trợ | Spark, DuckDB (delta-rs) | Spark, Trino, Flink | Spark, Flink |
| Học ngắn 1 tuần | ✅ Dễ nhất | Trung bình | Khó nhất |
| Time travel + RESTORE | ✅ `RESTORE VERSION AS OF` | ✅ snapshot id | Có nhưng phức tạp |
| Lab repo đã dùng sẵn | ✅ NB1–NB4 dùng Delta | ❌ Phải tự setup | ❌ Phải tự setup |
| Hidden partitioning | ❌ Cần explicit | ✅ `days(ts)` | ❌ |

**Quyết định: Delta Lake.**  
Constraint quyết định: (1) đã quen từ NB1–NB4 nên không tốn thời gian re-learn, (2) workload "append chunk + MERGE re-embed vector" hợp Delta, (3) RESTORE VERSION AS OF là tính năng thiết yếu khi embedding bug xảy ra.  
Trade-off chấp nhận: mất hidden partitioning của Iceberg → phải explicit partition theo `embed_date`.

---

### 2.2 Vector Storage: Delta ARRAY<FLOAT> vs tách Vector DB

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| Store vector trong Delta (`ARRAY<FLOAT>`) | Single source of truth, lineage đầy đủ, time travel native, không cần sync 2 hệ thống | ANN search chậm hơn vector DB; không có HNSW index |
| Tách: Delta metadata + Qdrant/Weaviate vectors | ANN nhanh hơn (HNSW), purpose-built | Lineage phải tự maintain; 2 hệ thống drift; sync bug nguy hiểm |
| pgvector trong PostgreSQL | Quen thuộc, ACID | Không phải lakehouse; không scale analytics |

**Quyết định: Store vector trong Delta (`ARRAY<FLOAT>`).**  
Lý do: Challenge yêu cầu kiến trúc Lakehouse, không phải production vector search system. Mục tiêu là observability và analytics trên embedding data, không phải millisecond ANN search. DuckDB + VSS extension đủ cho Gold analytics layer.  
Trade-off chấp nhận: không có HNSW index → kNN trong Gold chạy brute-force, acceptable với Gold < 2GB.

---

### 2.3 Compute Engine: Spark Micro-batch vs Flink Streaming

| Tiêu chí | Spark Structured Streaming (5-min micro-batch) | Apache Flink (true streaming) |
|---|---|---|
| SLA freshness | 5 phút — đủ với SLA 1 giờ | < 1s — overkill |
| Ops complexity | Thấp (1 Spark job) | Cao (Flink cluster, checkpointing, backpressure) |
| Cost | Baseline | ~4× vì cluster luôn chạy |
| Lab familiarity | ✅ Đã dùng trong NB3 | ❌ Chưa học |

**Quyết định: Spark Structured Streaming (5-min micro-batch).**

---

### 2.4 Serve Layer: DuckDB vs Trino Cluster

Gold mart size ước tính: < 2GB. DuckDB single-node trả kết quả < 2s với VSS extension. Trino cần tối thiểu 3 node, ops phức tạp hơn, không justified với volume này.  
**Quyết định: DuckDB + VSS extension.**  
Future work khi Gold > 50GB: migrate sang Trino hoặc Spark SQL.

---

## 3. Architecture Tổng Thể

### 3.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  SOURCE LAYER                                                        │
│  PDF docs ──┐                                                        │
│  PNG images ├──► Upload SDK ──► Kafka: doc.ingest.v1                │
│  Audio files┘                                                        │
│                                                                      │
│  RAG Retrieval SDK ──────────► Kafka: retrieval.events.v1           │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  (S3: s3://lh-rag-obs/bronze/)                        │
│  Delta — raw, immutable, partition by ingest_date                   │
│  bronze.raw_documents     — file metadata + S3 path                 │
│  bronze.raw_retrieval_logs — JSON payload nguyên gốc                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  Spark Structured Streaming (5-min batch)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SILVER LAYER  (S3: s3://lh-rag-obs/silver/)                        │
│  Delta — cleaned, normalized, Z-ORDER by (doc_id, modality)        │
│  silver.document_chunks   — text chunks sau chunking pipeline       │
│  silver.embeddings        — ARRAY<FLOAT> 1536 dims + cost tracking  │
│  silver.retrieval_events  — dedup, parsed, joined với chunk_id      │
│                                                                      │
│  MERGE pattern: khi re-embed → upsert vector vào silver.embeddings  │
│  Time travel: xem embedding trước khi re-index                      │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  dbt + DuckDB (daily cron 02:00)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GOLD LAYER  (S3: s3://lh-rag-obs/gold/)                            │
│  Delta — aggregated KPI marts, rebuild full daily                   │
│  gold.embedding_freshness_kpi — stale chunk detection               │
│  gold.retrieval_quality_kpi   — precision@5, MRR, zero-hit rate     │
│  gold.embedding_cost_kpi      — cost/ngày theo model + modality     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SERVE LAYER                                                         │
│  DuckDB (VSS extension) ──► Metabase dashboard                      │
│  OpenLineage ──► Marquez UI  (full DAG trace)                       │
│  Great Expectations         (quality gate tại mỗi layer boundary)   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Storage Paths

```
s3://lh-rag-obs/
├── bronze/
│   ├── raw_documents/          # partitioned by ingest_date
│   └── raw_retrieval_logs/     # partitioned by ingest_date
├── silver/
│   ├── document_chunks/        # partitioned by chunk_date
│   ├── embeddings/             # partitioned by embed_date
│   └── retrieval_events/       # partitioned by event_date
└── gold/
    ├── embedding_freshness_kpi/
    ├── retrieval_quality_kpi/
    └── embedding_cost_kpi/
```

### 3.3 Catalog

- **Production:** AWS Glue Data Catalog — single source of truth, cross-engine schema discovery
- **Development/Local:** Lakekeeper REST Catalog (chạy Docker) — compatible với Iceberg REST spec, dùng được với delta-rs
- **Trade-off:** Vendor lock vào AWS Glue, acceptable ở MVP stage

---

## 4. Schema Chi Tiết

### 4.1 Bronze Layer

#### `bronze.raw_documents`

```sql
CREATE TABLE bronze.raw_documents (
  doc_id            STRING       NOT NULL,  -- UUIDv7 từ upload SDK
  raw_s3_path       STRING       NOT NULL,  -- s3://... path tới file gốc
  doc_type          STRING       NOT NULL,  -- "pdf" | "image" | "audio"
  file_size_bytes   BIGINT,
  source_system     STRING,                 -- "confluence" | "sharepoint" | "manual_upload"
  tenant_id         STRING       NOT NULL,  -- multi-tenant
  ingest_ts         TIMESTAMP    NOT NULL,
  ingest_date       DATE         NOT NULL,  -- PARTITION COLUMN
  doc_hash          STRING       NOT NULL,  -- SHA-256 của file — detect update/duplicate
  schema_version    STRING       NOT NULL   -- "v1" — bump khi breaking change
)
USING DELTA
PARTITIONED BY (ingest_date)
TBLPROPERTIES (
  'delta.logRetentionDuration'        = '30 days',
  'delta.deletedFileRetentionDuration' = '30 days'
);
```

**Lý do thiết kế:**
- `raw_s3_path` thay vì store binary blob trong Delta: Delta không optimize cho binary large objects. File gốc nằm S3, Delta chỉ store metadata + path.
- `doc_hash` (SHA-256): cơ chế phát hiện document bị update. Khi hash thay đổi → trigger re-chunking + re-embedding pipeline.
- `schema_version`: khi producer SDK thay đổi format → bump "v1" → "v2", consumer route sang parser khác, tránh silent corruption.
- Bronze rule: không parse, không transform, không filter. Nếu một field bị null hoặc format lạ, Bronze vẫn nhận — Silver mới reject.

---

#### `bronze.raw_retrieval_logs`

```sql
CREATE TABLE bronze.raw_retrieval_logs (
  log_id          STRING       NOT NULL,  -- UUIDv7
  raw_payload     STRING       NOT NULL,  -- JSON string nguyên gốc từ retrieval SDK
  tenant_id       STRING       NOT NULL,
  ingest_ts       TIMESTAMP    NOT NULL,
  ingest_date     DATE         NOT NULL,  -- PARTITION COLUMN
  kafka_offset    BIGINT       NOT NULL,  -- dedup khi producer replay
  schema_version  STRING       NOT NULL
)
USING DELTA
PARTITIONED BY (ingest_date)
TBLPROPERTIES (
  'delta.logRetentionDuration'        = '7 days',
  'delta.deletedFileRetentionDuration' = '7 days'
);
```

**Lý do giữ `kafka_offset`:** khi Kafka topic bị lỗi và producer replay, dùng offset để dedup chính xác tại Silver, không bị double-count retrieval event.

---

### 4.2 Silver Layer

#### `silver.document_chunks`

```sql
CREATE TABLE silver.document_chunks (
  chunk_id            STRING       NOT NULL,  -- UUIDv7
  doc_id              STRING       NOT NULL,  -- FK → bronze.raw_documents.doc_id
  doc_version_hash    STRING       NOT NULL,  -- SHA-256 tại thời điểm chunk được tạo
  chunk_index         INT          NOT NULL,  -- vị trí chunk trong document (0-based)
  chunk_text          STRING       NOT NULL,  -- text sau clean: strip HTML, normalize unicode
  modality            STRING       NOT NULL,  -- "text" | "image_caption" | "audio_transcript"
  char_count          INT,
  token_estimate      INT,                    -- tiktoken estimate
  tenant_id           STRING       NOT NULL,
  chunk_ts            TIMESTAMP    NOT NULL,
  chunk_date          DATE         NOT NULL,  -- PARTITION COLUMN
  embedding_id        STRING,                 -- NULL cho đến khi embedding job chạy
  -- Lineage
  bronze_doc_id       STRING       NOT NULL,  -- back-pointer tới Bronze
  silver_run_id       STRING       NOT NULL   -- dbt run ID tạo row này
)
USING DELTA
PARTITIONED BY (chunk_date);

OPTIMIZE silver.document_chunks ZORDER BY (doc_id, modality);
```

**Lý do Z-ORDER `(doc_id, modality)`:** query pattern phổ biến nhất là "tất cả chunk của document X theo modality Y" — Z-ORDER co-locate 2 cột này, file pruning > 10× theo lab NB2.

**Lý do giữ `doc_version_hash` tại chunk level:** stale detection logic là `doc_version_hash != current doc_hash`. Nếu chỉ giữ hash ở Bronze, mỗi lần check freshness phải join qua 2 table. Denormalize hash về Silver chunk để query freshness KPI O(1).

---

#### `silver.embeddings`

```sql
CREATE TABLE silver.embeddings (
  embedding_id          STRING       NOT NULL,  -- UUIDv7
  chunk_id              STRING       NOT NULL,  -- FK → silver.document_chunks
  doc_id                STRING       NOT NULL,  -- denorm để tránh extra join
  tenant_id             STRING       NOT NULL,
  vector                ARRAY<FLOAT> NOT NULL,  -- 1536 dims (text-embed-3-large)
  vector_norm           FLOAT,                  -- pre-computed L2 norm cho cosine sim
  embedding_model       STRING       NOT NULL,  -- "text-embed-3-large" | "clip-vit-large"
  embed_ts              TIMESTAMP    NOT NULL,
  embed_date            DATE         NOT NULL,  -- PARTITION COLUMN
  prompt_tokens         INT,
  cost_usd              DECIMAL(10,8),
  -- Eval columns — đến trễ qua MERGE từ retrieval analytics pipeline
  retrieval_hit_count   INT,                    -- bao nhiêu lần chunk này được trả về
  avg_relevance_score   FLOAT,                  -- cosine similarity trung bình khi được chọn
  eval_updated_ts       TIMESTAMP,
  -- Lineage
  chunk_silver_run_id   STRING                  -- run ID tạo chunk tương ứng
)
USING DELTA
PARTITIONED BY (embed_date);

OPTIMIZE silver.embeddings ZORDER BY (doc_id, embedding_model);
```

**MERGE pattern — khi document bị update và re-embed:**

```sql
-- Trigger: doc_hash mới != doc_version_hash trong silver.document_chunks
-- Pipeline: re-chunk → re-embed → push vào staging.reembed_results
-- Sau đó MERGE:

MERGE INTO silver.embeddings AS target
USING staging.reembed_results AS source
  ON target.chunk_id = source.chunk_id
WHEN MATCHED THEN
  UPDATE SET
    target.vector              = source.vector,
    target.vector_norm         = source.vector_norm,
    target.embedding_model     = source.embedding_model,
    target.embed_ts            = current_timestamp(),
    target.prompt_tokens       = source.prompt_tokens,
    target.cost_usd            = source.cost_usd
WHEN NOT MATCHED THEN
  INSERT *;

-- Sau MERGE: _delta_log ghi version mới
-- Time travel: xem embedding trước khi re-index
-- SELECT * FROM silver.embeddings VERSION AS OF 42 WHERE chunk_id = 'abc';
```

**Lý do `vector_norm` pre-computed:** cosine similarity = dot_product / (norm_a × norm_b). Nếu tính norm on-the-fly mỗi query sẽ O(D) cho mỗi vector trong brute-force scan. Pre-compute 1 lần khi embed, tái dùng mãi mãi.

---

#### `silver.retrieval_events`

```sql
CREATE TABLE silver.retrieval_events (
  event_id            STRING       NOT NULL,  -- parse từ raw_payload
  query_id            STRING       NOT NULL,
  user_id             STRING,                 -- hash PII trước khi vào Silver (Decree-13)
  tenant_id           STRING       NOT NULL,
  query_text_hash     STRING,                 -- SHA-256 của query text (không store raw text)
  query_modality      STRING,                 -- "text" | "image" | "audio"
  retrieved_chunk_ids ARRAY<STRING>,          -- top-k chunks trả về
  relevance_scores    ARRAY<FLOAT>,           -- cosine sim tương ứng
  top1_chunk_id       STRING,                 -- denorm cho fast aggregation
  top1_score          FLOAT,
  is_zero_hit         BOOLEAN,                -- true nếu top1_score < 0.3
  latency_ms          INT,
  event_ts            TIMESTAMP    NOT NULL,
  event_date          DATE         NOT NULL,  -- PARTITION COLUMN
  -- Lineage
  bronze_log_id       STRING       NOT NULL,
  bronze_kafka_offset BIGINT       NOT NULL   -- dedup reference
)
USING DELTA
PARTITIONED BY (event_date);
```

**Lý do hash `user_id` và `query_text_hash`:** Decree-13/2023/NĐ-CP yêu cầu xử lý data cá nhân. Query text có thể chứa PII (tên, số điện thoại). Hash SHA-256 tại Silver → irreversible → compliance. Trade-off: không debug được raw query khi cần, phải trace ngược Bronze qua `bronze_log_id`.

---

### 4.3 Gold Layer

#### `gold.embedding_freshness_kpi`

```sql
CREATE TABLE gold.embedding_freshness_kpi (
  report_date           DATE         NOT NULL,
  tenant_id             STRING       NOT NULL,
  modality              STRING       NOT NULL,
  total_chunk_count     INT,
  stale_chunk_count     INT,         -- doc_version_hash != latest doc_hash
  fresh_chunk_count     INT,
  freshness_ratio       FLOAT,       -- fresh / total
  oldest_stale_embed_ts TIMESTAMP,   -- embed_ts của chunk stale lâu nhất
  avg_days_since_embed  FLOAT,
  pending_reembed_cost_estimate_usd DECIMAL(10,4)
                                     -- ước tính cost nếu re-embed toàn bộ stale
)
USING DELTA;
```

---

#### `gold.retrieval_quality_kpi`

```sql
CREATE TABLE gold.retrieval_quality_kpi (
  report_date           DATE         NOT NULL,
  tenant_id             STRING       NOT NULL,
  query_modality        STRING       NOT NULL,
  embedding_model       STRING       NOT NULL,
  total_queries         INT,
  zero_hit_count        INT,         -- top1_score < 0.3
  zero_hit_rate         FLOAT,
  avg_top1_score        FLOAT,
  p50_top1_score        FLOAT,
  p95_top1_score        FLOAT,
  precision_at_5        FLOAT,       -- tỉ lệ relevant trong top 5 (dùng user feedback)
  mrr                   FLOAT        -- Mean Reciprocal Rank
)
USING DELTA;
```

---

#### `gold.embedding_cost_kpi`

```sql
CREATE TABLE gold.embedding_cost_kpi (
  report_date       DATE         NOT NULL,
  tenant_id         STRING       NOT NULL,
  embedding_model   STRING       NOT NULL,
  modality          STRING       NOT NULL,
  embed_request_count INT,
  total_tokens      BIGINT,
  total_cost_usd    DECIMAL(10,4),
  avg_cost_per_chunk DECIMAL(10,6),
  reembed_count     INT          -- re-embed do document update
)
USING DELTA;
```

---

## 5. Production Ops Decisions

### 5.1 Compaction Schedule

```
bronze.*         OPTIMIZE daily 01:00 (không Z-ORDER — chỉ cần small file consolidation)
silver.chunks    OPTIMIZE + ZORDER BY (doc_id, modality) daily 02:00
silver.embeddings OPTIMIZE + ZORDER BY (doc_id, embedding_model) daily 02:30
silver.retrieval  OPTIMIZE + ZORDER BY (tenant_id, event_date) daily 02:45
gold.*            Rebuild full daily 03:00 (idempotent — DROP + INSERT OVERWRITE)
```

```
VACUUM bronze.*          RETAIN 30 DAYS  (compliance: 30 ngày raw)
VACUUM silver.embeddings RETAIN 30 DAYS  (audit re-index history)
VACUUM silver.chunks     RETAIN 7 DAYS
VACUUM silver.retrieval  RETAIN 7 DAYS
VACUUM gold.*            RETAIN 3 DAYS
```

**Lý do giữ Silver embeddings 30 ngày:** khi có embedding bug, cần so sánh `VERSION AS OF before_reembed` vs `VERSION AS OF after_reembed`. VACUUM quá sớm sẽ xóa mất history trước khi team kịp investigate.

---

### 5.2 Stale Detection Pipeline

Chạy hourly bằng Spark job nhỏ:

```
1. Read bronze.raw_documents (doc_id, doc_hash) — version hiện tại
2. Read silver.document_chunks (chunk_id, doc_id, doc_version_hash)
3. JOIN ON doc_id WHERE doc_hash != doc_version_hash
4. Write vào staging.stale_chunks
5. Trigger re-chunking + re-embedding job cho các doc_id bị stale
6. Sau khi embed xong → MERGE vào silver.embeddings
```

---

### 5.3 Data Contracts

- `schema_version` pinned tại producer SDK (Upload Service)
- Breaking change → bump "v1" → "v2" → dual-write 2 tuần → deprecate v1
- Great Expectations chạy tại 2 gate:
  - **Gate 1 (Bronze → Silver):** `doc_hash` không null, `vector` length == 1536, `chunk_text` length > 10
  - **Gate 2 (Silver → Gold):** `freshness_ratio` trong khoảng [0,1], `zero_hit_rate` < 0.5 (alert nếu vượt)

---

### 5.4 Lineage

- OpenLineage emitter cấu hình trong Spark job (Bronze→Silver) và dbt (Silver→Gold)
- Marquez UI: trace 1 row `gold.retrieval_quality_kpi` ngược về Kafka offset gốc
- Mỗi row Silver có `silver_run_id` → join với `dbt_runs` table để biết batch nào sinh ra row lỗi

---

### 5.5 FinOps Tiering

| Layer | Tier mặc định | Lifecycle rule |
|---|---|---|
| Bronze | S3 Standard 30d | → S3 IA 30d → S3 Glacier 1 năm |
| Silver | S3 Standard 90d | → S3 IA |
| Gold | S3 Standard mãi mãi | Nhỏ, hot, query thường xuyên |

---

### 5.6 Alerting

| Điều kiện | Action |
|---|---|
| `stale_chunk_count / total_chunk_count > 0.2` | Alert: hơn 20% chunks đang stale |
| `zero_hit_rate > 0.3` trong 24h | Alert: retrieval quality degraded |
| Silver row count < Bronze × 0.9 | Alert: chunking pipeline đang drop nhiều records |
| Daily embedding cost > baseline × 1.5 | Alert: chi phí bất thường |

---

### 5.7 PII & Compliance (Decree-13/2023/NĐ-CP)

- `user_id` trong `silver.retrieval_events`: hash SHA-256 tại ingestion vào Silver. Raw `user_id` chỉ tồn tại trong Bronze (JSON payload) với retention 7 ngày.
- `query_text`: không store raw. Store SHA-256 hash. Nếu cần debug, trace ngược `bronze_log_id` trong window 7 ngày.
- **Right-to-be-forgotten:**
  ```sql
  DELETE FROM silver.document_chunks     WHERE tenant_id = 'X';
  DELETE FROM silver.embeddings          WHERE tenant_id = 'X';
  DELETE FROM silver.retrieval_events    WHERE tenant_id = 'X';
  VACUUM silver.document_chunks          RETAIN 0 HOURS;  -- tắt time travel cho tenant này
  VACUUM silver.embeddings               RETAIN 0 HOURS;
  VACUUM silver.retrieval_events         RETAIN 0 HOURS;
  ```
  **Cảnh báo:** `VACUUM RETAIN 0 HOURS` xóa tất cả file cũ không còn trong Delta log hiện tại, bao gồm time travel history. Phải confirm với compliance team trước khi chạy.
- Audit log: ghi lại ai/khi nào READ Gold tables (AWS CloudTrail + Glue data access log).

---

## 6. Trade-offs (Cái Cố Ý Từ Chối)

| Quyết định chọn | Cái từ chối | Lý do |
|---|---|---|
| Delta Lake | Iceberg hidden partitioning | 1 tuần không đủ học Iceberg production; đã quen Delta từ lab |
| Store vector trong Delta `ARRAY<FLOAT>` | Qdrant / Weaviate / pgvector | Challenge yêu cầu Lakehouse; mục tiêu là observability, không phải ANN latency |
| DuckDB + VSS | Trino cluster | Gold < 2GB; DuckDB < 2s; Trino cần 3+ node, ops phức tạp không justified |
| Spark 5-min micro-batch | Flink true streaming | SLA freshness 1 giờ; micro-batch đủ; Flink 4× cost Spark |
| Hash PII tại Silver | Encrypt-at-rest only | Hash irreversible → an toàn hơn. Trade-off: mất khả năng decrypt khi debug |
| Gold rebuild full daily | Incremental update Gold | Idempotent đơn giản hơn. Trade-off: tốn compute hơn nhưng < 5 phút nên OK |
| 1 catalog (AWS Glue) | Multi-catalog | Single source of truth. Trade-off: vendor-lock AWS, chấp nhận ở MVP |
| Explicit partition `embed_date` | Iceberg `days(embed_ts)` | Hệ quả của việc chọn Delta thay Iceberg |

---

## 7. Failure Scenarios

### Scenario 1: Embedding provider timeout — OpenAI API down

**Hậu quả:** Embedding job fail, chunk_id trong `silver.document_chunks` có `embedding_id = NULL`.  
**Giải pháp:** Bronze vẫn có raw document nguyên vẹn tại `raw_s3_path`. Silver chunks đã có `chunk_text`. Re-run embedding job từ Silver (không cần re-chunk từ Bronze). Job idempotent: check `embedding_id IS NULL` trước khi gửi API.  
**Dedup:** `embedding_id` là UUIDv7 generate tại client trước khi gửi API → nếu API trả về timeout nhưng thực ra đã xử lý, MERGE bằng `chunk_id` sẽ không tạo duplicate.

---

### Scenario 2: Re-embed bug — vector sai do model version thay đổi

**Hậu quả:** 5,000 rows trong `silver.embeddings` có vector generated bởi model version lỗi, dẫn đến retrieval quality drop.  
**Giải pháp:**
```sql
-- Xác định version trước khi re-embed job chạy
DESCRIBE HISTORY silver.embeddings;  -- tìm version N (trước khi MERGE)

-- Rollback
RESTORE TABLE silver.embeddings TO VERSION AS OF N;
```
Delta time travel cứu toàn bộ mà không cần backup snapshot thủ công.

---

### Scenario 3: User yêu cầu xóa dữ liệu (Decree-13 right-to-be-forgotten)

**Hậu quả:** Phải xóa physically (không soft-delete) toàn bộ data của tenant X trong vòng 72 giờ theo luật.  
**Giải pháp:**
1. `DELETE FROM silver.* WHERE tenant_id = 'X'` — xóa logical khỏi Delta log
2. `VACUUM silver.* RETAIN 0 HOURS` — xóa physical Parquet files
3. Xóa Bronze `raw_s3_path` files trực tiếp trong S3 (Bronze store path, không store blob)
4. Ghi audit log: timestamp, operator, tenant_id đã xóa

**Cảnh báo:** Sau bước 2, time travel history của tenant X không còn. Cần xác nhận compliance trước khi chạy `RETAIN 0 HOURS`.

---

### Scenario 4: Retrieval quality drop đột ngột — CEO panic

**Hậu quả:** `zero_hit_rate` tăng từ 8% lên 45% trong 24 giờ.  
**Giải pháp:**
1. Query `gold.retrieval_quality_kpi` → xác định modality và tenant bị ảnh hưởng
2. Query `gold.embedding_freshness_kpi` → check `freshness_ratio` có giảm không (stale embeddings?)
3. Nếu stale: check `staging.stale_chunks` → có batch re-embed nào fail không
4. Nếu không stale: trace `silver.retrieval_events` → check `avg_relevance_score` distribution thay đổi → có thể embedding model đã thay đổi silently (check `embedding_model` field)
5. Rollback model: `RESTORE TABLE silver.embeddings TO VERSION AS OF N` về trước khi model thay đổi

---

## 8. Implementation Roadmap (1 Tuần)

| Day | Task | Deliverable | Risk |
|---|---|---|---|
| D1 | Setup MinIO local + Kafka + Upload SDK fake docs (PDF/PNG/audio) | 2,000 fake docs đẩy được vào Kafka `doc.ingest.v1` | Low |
| D2 | Bronze ingest job (Spark Structured Streaming) + Chunking pipeline (LangChain RecursiveCharacterTextSplitter) | `bronze.raw_documents` có data; 50k chunks trong Silver | Low |
| D3 | Embedding job — gọi OpenAI API (mock với numpy nếu không có key) + MERGE pattern | `silver.embeddings` populated; MERGE re-embed hoạt động | Med |
| D4 | Silver retrieval events ingest + dbt Gold models | 3 Gold KPI marts; query < 3s | Med |
| D5 | Great Expectations gates + Stale detection pipeline | Alert bắn khi freshness_ratio < 0.8; GE test fail có log | **High** |
| D6 | OpenLineage + Marquez integration + Metabase dashboard | Marquez UI show full DAG; dashboard 3 charts | Med |
| D7 | Buffer day: PII hash audit, VACUUM test, write-up, slide | `architecture.md` + slide deck + VACUUM compliance demo | Low |

**Ghi chú D5 là risk cao nhất:** Great Expectations version phụ thuộc vào Spark version. OpenLineage emitter có nhiều quirks tùy version Spark. Buffer D7 để absorb nếu D5–D6 trễ.

---

## 9. Demo Numbers (Dự Kiến)

| Metric | Giá trị | Ghi chú |
|---|---|---|
| Total documents processed | 2,000 | 3 modalities |
| Total chunks | 50,000 | avg 25 chunks/doc |
| Silver embeddings | 50,000 rows × 1536 floats = ~300MB | manageable với DuckDB |
| Stale detection latency | < 1 phút | Spark job hourly |
| Gold query: freshness_ratio by tenant | 47ms | DuckDB on Gold |
| Gold query: daily cost breakdown | 62ms | |
| Z-ORDER file pruning | ~12× | theo lab NB2 baseline |
| MERGE re-embed 1,000 chunks | < 30s | Spark MERGE operation |
| Time travel rollback | < 5s | `RESTORE VERSION AS OF N` |

**Sample query:**
```sql
-- "Document nào đang stale nhất của tenant acme?"
SELECT doc_id, modality, oldest_stale_embed_ts,
       pending_reembed_cost_estimate_usd
FROM gold.embedding_freshness_kpi
WHERE tenant_id = 'acme'
  AND report_date = current_date()
  AND stale_chunk_count > 0
ORDER BY oldest_stale_embed_ts ASC
LIMIT 10;
-- → 47ms, 10 rows
```

---

## 10. Lessons Learned

**Cái đúng:**
- Viết problem statement và success criteria trước khi vẽ kiến trúc → tránh over-engineer (không dùng Flink khi 5-min batch đủ)
- Decision matrix theo constraint (1 tuần, quen Delta, Gold < 2GB) thay vì theo feature set của tool
- `doc_version_hash` denorm về Silver chunk → stale detection O(1) không cần join Bronze
- Buffer D7 → kịp deadline dù D5 hoặc D6 trễ

**Cái cần làm khác:**
- Lúc đầu định thêm Qdrant cho vector search — phí 1 ngày research, cuối cùng không cần vì Challenge yêu cầu Lakehouse không yêu cầu ANN
- Quên pre-compute `vector_norm` đến lúc viết Gold query cosine sim mới nhớ — phải backfill Silver

**Future work khi scale 10×:**
- Migrate embedding storage sang Qdrant hoặc pgvector khi cần ANN latency < 100ms
- Migrate Delta → Iceberg để có hidden partitioning `days(embed_ts)` — bỏ được explicit `embed_date` partition
- Flink real-time thay Spark micro-batch khi SLA freshness xuống < 1 phút
- Multi-region (SGP + HCM) với active-active cho compliance dữ liệu trong nước

---

## 11. References

- Day 18 deck (Track 2) — VinUni AI20k
- Lab repo: `VinUni-AI20k/Day18-Track2-Lakehouse-Lab`
- `DAY18-Lakehouse-DETAILED-Walkthrough.md`
- Delta Lake documentation — `delta.io/docs`
- OpenLineage specification — `openlineage.io`
- Great Expectations documentation — `greatexpectations.io`
- Decree 13/2023/NĐ-CP — Nghị định về bảo vệ dữ liệu cá nhân
- DuckDB VSS extension — `duckdb.org/docs/extensions/vss`

---

