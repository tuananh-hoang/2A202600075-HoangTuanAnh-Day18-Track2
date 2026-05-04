"""
data_generator.py — Tạo dữ liệu giả cho Multimodal RAG Lakehouse PoC

Sinh:
  - Danh sách documents (PDF/image/audio) với metadata
  - Retrieval event logs (JSON payload)
  - Embedding vectors (numpy random, normalized)
  - Stale document batches (simulate document update)
"""

import uuid
import hashlib
import json
import random
import numpy as np
import pandas as pd
import pyarrow as pa
from datetime import datetime, timedelta
from faker import Faker

import config

fake = Faker("vi_VN")
rng  = np.random.default_rng(config.SEED)
random.seed(config.SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_uuid() -> str:
    return str(uuid.uuid4())

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def rand_vector(dim: int = config.VECTOR_DIM) -> list[float]:
    """Tạo unit vector ngẫu nhiên (L2-normalized)."""
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()

def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

def now_ts() -> datetime:
    return datetime.utcnow()

def days_ago(n: int) -> datetime:
    return datetime.utcnow() - timedelta(days=n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Documents
# ─────────────────────────────────────────────────────────────────────────────

def gen_raw_documents() -> pa.Table:
    """Sinh 120 documents (40 mỗi modality: text/image/audio)."""
    rows = []
    per_modality = config.N_DOCS // len(config.MODALITIES)

    for modality_idx, modality in enumerate(config.MODALITIES):
        ext = {"text": "pdf", "image_caption": "png", "audio_transcript": "mp3"}[modality]
        for i in range(per_modality):
            doc_id   = make_uuid()
            filename = f"{modality}_{i:04d}.{ext}"
            content  = f"{fake.sentence()} {fake.paragraph()}"
            doc_hash = sha256(content + doc_id)
            tenant   = config.TENANTS[i % len(config.TENANTS)]
            ingest_dt = days_ago(random.randint(1, 30))
            rows.append({
                "doc_id":          doc_id,
                "raw_s3_path":     f"s3://lh-rag-obs/raw/{tenant}/{modality}/{filename}",
                "doc_type":        ext,
                "modality":        modality,
                "file_size_bytes": random.randint(10_000, 5_000_000),
                "source_system":   random.choice(["confluence", "sharepoint", "manual_upload"]),
                "tenant_id":       tenant,
                "ingest_ts":       ingest_dt,
                "ingest_date":     ingest_dt.date(),
                "doc_hash":        doc_hash,
                "schema_version":  "v1",
            })

    return _to_arrow_docs(rows)


def _to_arrow_docs(rows: list[dict]) -> pa.Table:
    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, preserve_index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Raw Retrieval Logs
# ─────────────────────────────────────────────────────────────────────────────

def gen_raw_retrieval_logs(doc_ids: list[str]) -> pa.Table:
    """Sinh N_RETRIEVAL_EVENTS retrieval log events dưới dạng JSON payload."""
    rows = []
    for i in range(config.N_RETRIEVAL_EVENTS):
        tenant    = random.choice(config.TENANTS)
        query_mod = random.choice(config.MODALITIES)
        log_id    = make_uuid()
        ingest_dt = days_ago(random.randint(0, 14))

        payload = {
            "event_id":      make_uuid(),
            "query_id":      make_uuid(),
            "user_id":       f"user_{random.randint(1, 50):04d}",
            "tenant_id":     tenant,
            "query_text":    fake.sentence(),
            "query_modality": query_mod,
            "retrieved_doc_ids": random.sample(doc_ids, k=min(5, len(doc_ids))),
            "latency_ms":    random.randint(50, 800),
            "ts":            ingest_dt.isoformat(),
        }

        rows.append({
            "log_id":         log_id,
            "raw_payload":    json.dumps(payload, ensure_ascii=False),
            "tenant_id":      tenant,
            "ingest_ts":      ingest_dt,
            "ingest_date":    ingest_dt.date(),
            "kafka_offset":   i + 1_000_000,
            "schema_version": "v1",
        })

    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, preserve_index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Document Chunks
# ─────────────────────────────────────────────────────────────────────────────

def gen_document_chunks(docs_df: pd.DataFrame, silver_run_id: str) -> pa.Table:
    """
    Sinh chunks từ danh sách documents.
    Mỗi document → N_CHUNKS_PER_DOC chunks.
    """
    rows = []
    for _, doc in docs_df.iterrows():
        for chunk_idx in range(config.N_CHUNKS_PER_DOC):
            chunk_id   = make_uuid()
            chunk_text = f"{fake.paragraph()} [chunk {chunk_idx} of {doc['doc_id'][:8]}]"
            chunk_dt   = doc["ingest_ts"] + timedelta(minutes=random.randint(1, 10))
            rows.append({
                "chunk_id":          chunk_id,
                "doc_id":            doc["doc_id"],
                "doc_version_hash":  doc["doc_hash"],     # lưu hash lúc chunk được tạo
                "chunk_index":       chunk_idx,
                "chunk_text":        chunk_text,
                "modality":          doc["modality"],
                "char_count":        len(chunk_text),
                "token_estimate":    len(chunk_text) // 4,
                "tenant_id":         doc["tenant_id"],
                "chunk_ts":          chunk_dt,
                "chunk_date":        chunk_dt.date(),
                "embedding_id":      None,               # NULL cho đến khi embed xong
                "bronze_doc_id":     doc["doc_id"],
                "silver_run_id":     silver_run_id,
            })

    df = pd.DataFrame(rows)

    # Định nghĩa schema tường minh để tránh PyArrow infer type Null
    # cho các cột nullable — Delta Lake không chấp nhận type Null
    schema = pa.schema([
        pa.field("chunk_id",         pa.string()),
        pa.field("doc_id",           pa.string()),
        pa.field("doc_version_hash", pa.string()),
        pa.field("chunk_index",      pa.int32()),
        pa.field("chunk_text",       pa.string()),
        pa.field("modality",         pa.string()),
        pa.field("char_count",       pa.int32()),
        pa.field("token_estimate",   pa.int32()),
        pa.field("tenant_id",        pa.string()),
        pa.field("chunk_ts",         pa.timestamp("us")),
        pa.field("chunk_date",       pa.date32()),
        pa.field("embedding_id",     pa.string()),   # nullable string, không phải Null type
        pa.field("bronze_doc_id",    pa.string()),
        pa.field("silver_run_id",    pa.string()),
    ])
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def gen_embeddings(chunks_df: pd.DataFrame) -> pa.Table:
    """Sinh mock embedding cho mỗi chunk."""
    rows  = []
    vecs  = []

    for _, chunk in chunks_df.iterrows():
        embedding_id = make_uuid()
        vector       = rand_vector(config.VECTOR_DIM)
        vector_norm  = float(np.linalg.norm(vector))
        tokens        = chunk["token_estimate"]
        cost          = round(tokens / 1000 * config.COST_PER_1K_TOKENS, 8)
        embed_dt      = chunk["chunk_ts"] + timedelta(minutes=random.randint(1, 5))

        rows.append({
            "embedding_id":    embedding_id,
            "chunk_id":        chunk["chunk_id"],
            "doc_id":          chunk["doc_id"],
            "tenant_id":       chunk["tenant_id"],
            "vector_norm":     vector_norm,
            "embedding_model": config.EMBEDDING_MODEL,
            "embed_ts":        embed_dt,
            "embed_date":      embed_dt.date(),
            "prompt_tokens":   tokens,
            "cost_usd":        cost,
            # eval cols — NULL lúc đầu, được MERGE sau
            "retrieval_hit_count":   None,
            "avg_relevance_score":   None,
            "eval_updated_ts":       None,
            "chunk_silver_run_id":   chunk["silver_run_id"],
        })
        vecs.append(vector)

    df = pd.DataFrame(rows)

    schema = pa.schema([
        pa.field("embedding_id",          pa.string()),
        pa.field("chunk_id",              pa.string()),
        pa.field("doc_id",               pa.string()),
        pa.field("tenant_id",            pa.string()),
        pa.field("vector_norm",          pa.float64()),
        pa.field("embedding_model",      pa.string()),
        pa.field("embed_ts",             pa.timestamp("us")),
        pa.field("embed_date",           pa.date32()),
        pa.field("prompt_tokens",        pa.int32()),
        pa.field("cost_usd",             pa.float64()),
        pa.field("retrieval_hit_count",  pa.int32()),
        pa.field("avg_relevance_score",  pa.float64()),
        pa.field("eval_updated_ts",      pa.timestamp("us")),
        pa.field("chunk_silver_run_id",  pa.string()),
    ])
    base = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    vec_col = pa.array(vecs, type=pa.list_(pa.float32()))
    return base.append_column(
        pa.field("vector", pa.list_(pa.float32())), vec_col
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Retrieval Events (Silver — parsed từ Bronze)
# ─────────────────────────────────────────────────────────────────────────────

def gen_retrieval_events(bronze_logs_df: pd.DataFrame,
                         chunk_ids: list[str]) -> pa.Table:
    """
    Parse JSON payload từ Bronze → Silver retrieval events.
    Hash PII: user_id và query_text.
    """
    rows = []
    for _, log in bronze_logs_df.iterrows():
        payload = json.loads(log["raw_payload"])

        # Lấy top-k chunks ngẫu nhiên (simulate ANN search result)
        top_k = random.sample(chunk_ids, k=min(5, len(chunk_ids)))
        scores = sorted(
            [round(random.uniform(0.1, 0.99), 4) for _ in top_k],
            reverse=True
        )
        top1_score = scores[0]

        event_dt = datetime.fromisoformat(payload["ts"])
        rows.append({
            "event_id":             payload["event_id"],
            "query_id":             payload["query_id"],
            "user_id":              sha256(payload["user_id"]),    # PII hash
            "tenant_id":            payload["tenant_id"],
            "query_text_hash":      sha256(payload["query_text"]), # PII hash
            "query_modality":       payload["query_modality"],
            "retrieved_chunk_ids":  json.dumps(top_k),             # store as JSON string
            "relevance_scores":     json.dumps(scores),
            "top1_chunk_id":        top_k[0],
            "top1_score":           top1_score,
            "is_zero_hit":          top1_score < config.ZERO_HIT_THRESHOLD,
            "latency_ms":           payload["latency_ms"],
            "event_ts":             event_dt,
            "event_date":           event_dt.date(),
            "bronze_log_id":        log["log_id"],
            "bronze_kafka_offset":  log["kafka_offset"],
        })

    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, preserve_index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stale Document Update (simulate document bị update)
# ─────────────────────────────────────────────────────────────────────────────

def gen_stale_documents(docs_df: pd.DataFrame,
                        n_stale: int = 20) -> pd.DataFrame:
    """
    Simulate N document được update → doc_hash thay đổi.
    Trả về DataFrame các doc mới (updated docs với hash mới).
    """
    rng_local = np.random.default_rng(config.SEED + 99)
    sample    = docs_df.sample(n=n_stale, random_state=config.SEED).copy()
    # Tạo hash mới (simulate file content thay đổi)
    sample["doc_hash"] = sample["doc_id"].apply(
        lambda did: sha256(did + "_v2_updated_" + str(rng_local.integers(1_000_000)))
    )
    return sample


def gen_reembed_batch(stale_chunks_df: pd.DataFrame) -> pa.Table:
    """
    Tạo embedding mới cho các chunks thuộc stale documents.
    Đây là input cho MERGE operation.
    """
    rows = []
    vecs = []
    for _, chunk in stale_chunks_df.iterrows():
        new_embedding_id = make_uuid()
        vector           = rand_vector(config.VECTOR_DIM)   # vector mới
        vector_norm      = float(np.linalg.norm(vector))
        cost             = round(chunk["token_estimate"] / 1000 * config.COST_PER_1K_TOKENS, 8)
        embed_dt         = now_ts()

        rows.append({
            "embedding_id":    new_embedding_id,
            "chunk_id":        chunk["chunk_id"],
            "doc_id":          chunk["doc_id"],
            "tenant_id":       chunk["tenant_id"],
            "vector_norm":     vector_norm,
            "embedding_model": config.EMBEDDING_MODEL,
            "embed_ts":        embed_dt,
            "embed_date":      embed_dt.date(),
            "prompt_tokens":   chunk["token_estimate"],
            "cost_usd":        cost,
            "retrieval_hit_count":  None,
            "avg_relevance_score":  None,
            "eval_updated_ts":      None,
            "chunk_silver_run_id":  chunk["silver_run_id"],
        })
        vecs.append(vector)

    df = pd.DataFrame(rows)
    base = pa.Table.from_pandas(df, preserve_index=False)
    vec_col = pa.array(vecs, type=pa.list_(pa.float32()))
    return base.append_column(
        pa.field("vector", pa.list_(pa.float32())), vec_col
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Eval updates (MERGE vào Silver embeddings sau khi user click 👍/👎)
# ─────────────────────────────────────────────────────────────────────────────

def gen_eval_updates(embeddings_df: pd.DataFrame,
                     retrieval_df: pd.DataFrame) -> pa.Table:
    """
    Aggregate retrieval stats theo chunk_id → MERGE vào Silver embeddings.
    Simulate việc eval pipeline chạy daily và cập nhật hit_count + avg_score.
    """
    # Parse retrieved_chunk_ids từ JSON string
    all_hits = []
    for _, ev in retrieval_df.iterrows():
        chunk_ids = json.loads(ev["retrieved_chunk_ids"])
        scores    = json.loads(ev["relevance_scores"])
        for cid, sc in zip(chunk_ids, scores):
            all_hits.append({"chunk_id": cid, "score": sc})

    hits_df = pd.DataFrame(all_hits)
    if hits_df.empty:
        return pa.Table.from_pandas(pd.DataFrame(), preserve_index=False)

    agg = hits_df.groupby("chunk_id").agg(
        retrieval_hit_count=("score", "count"),
        avg_relevance_score=("score", "mean"),
    ).reset_index()

    # Chỉ update những chunk đã có embedding
    agg["eval_updated_ts"] = now_ts()

    # Join để lấy các cột cần cho MERGE
    merged = agg.merge(
        embeddings_df[["chunk_id", "embedding_id", "doc_id", "tenant_id",
                        "embedding_model", "embed_ts", "embed_date",
                        "prompt_tokens", "cost_usd", "vector_norm",
                        "chunk_silver_run_id"]],
        on="chunk_id", how="inner"
    )

    return pa.Table.from_pandas(merged, preserve_index=False)