"""
gold_layer.py — Xây dựng Gold KPI marts bằng DuckDB

DuckDB đọc Parquet files của Silver Delta tables (thông qua active file list),
tính toán các KPI, ghi kết quả ra Gold (Parquet + Delta).
"""

import os
import json
from datetime import datetime

import duckdb
import pandas as pd
import pyarrow as pa

import config
import lake_ops


def _get_conn(silver_chunks_files: list[str],
              silver_embed_files:  list[str],
              silver_retrieval_files: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Tạo DuckDB connection với 3 view trỏ vào Parquet files của Silver tables.
    Đây là cách đọc Delta table qua DuckDB khi không có delta extension:
    dùng DeltaTable.file_uris() để lấy đúng Parquet files đang active,
    tránh đọc Parquet cũ đã bị supersede bởi MERGE/OPTIMIZE.
    """
    conn = duckdb.connect(":memory:")

    def register(name: str, files: list[str]) -> None:
        if not files:
            return
        files_lit = ", ".join(f"'{f}'" for f in files)
        conn.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_parquet([{files_lit}])
        """)

    register("silver_chunks",    silver_chunks_files)
    register("silver_embeddings", silver_embed_files)
    register("silver_retrieval",  silver_retrieval_files)

    return conn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding Freshness KPI
# ─────────────────────────────────────────────────────────────────────────────

def build_freshness_kpi(conn: duckdb.DuckDBPyConnection,
                        current_doc_hashes: dict[str, str]) -> pd.DataFrame:
    """
    Phát hiện stale chunks: chunk.doc_version_hash != current doc_hash trong Bronze.
    current_doc_hashes: {doc_id -> latest_doc_hash}
    """
    # Inject current hashes vào DuckDB
    hash_rows = [{"doc_id": k, "current_hash": v}
                 for k, v in current_doc_hashes.items()]
    conn.register("current_hashes", pa.Table.from_pylist(hash_rows))

    return conn.execute("""
        WITH chunk_status AS (
            SELECT
                c.chunk_id,
                c.doc_id,
                c.tenant_id,
                c.modality,
                c.doc_version_hash,
                c.chunk_ts,
                h.current_hash,
                CASE WHEN c.doc_version_hash != h.current_hash
                     THEN TRUE ELSE FALSE END AS is_stale,
                e.embed_ts,
                e.cost_usd
            FROM silver_chunks c
            LEFT JOIN current_hashes h ON c.doc_id = h.doc_id
            LEFT JOIN silver_embeddings e ON c.chunk_id = e.chunk_id
        ),
        agg AS (
            SELECT
                CURRENT_DATE                         AS report_date,
                tenant_id,
                modality,
                COUNT(*)                             AS total_chunk_count,
                SUM(CASE WHEN is_stale THEN 1 ELSE 0 END) AS stale_chunk_count,
                SUM(CASE WHEN NOT is_stale THEN 1 ELSE 0 END) AS fresh_chunk_count,
                ROUND(
                    SUM(CASE WHEN NOT is_stale THEN 1.0 ELSE 0.0 END) / COUNT(*), 4
                )                                    AS freshness_ratio,
                MIN(CASE WHEN is_stale THEN embed_ts END) AS oldest_stale_embed_ts,
                ROUND(AVG(DATEDIFF('day', embed_ts, CURRENT_TIMESTAMP)), 2) AS avg_days_since_embed,
                ROUND(SUM(CASE WHEN is_stale THEN cost_usd ELSE 0 END), 6)
                                                     AS pending_reembed_cost_usd
            FROM chunk_status
            GROUP BY tenant_id, modality
        )
        SELECT * FROM agg
        ORDER BY tenant_id, modality
    """).fetchdf()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Quality KPI
# ─────────────────────────────────────────────────────────────────────────────

def build_quality_kpi(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        WITH base AS (
            SELECT
                CAST(event_date AS DATE)             AS report_date,
                tenant_id,
                query_modality,
                top1_score,
                is_zero_hit,
                latency_ms,
                top1_chunk_id
            FROM silver_retrieval
        ),
        agg AS (
            SELECT
                report_date,
                tenant_id,
                query_modality,
                COUNT(*)                             AS total_queries,
                SUM(CASE WHEN is_zero_hit THEN 1 ELSE 0 END) AS zero_hit_count,
                ROUND(
                    SUM(CASE WHEN is_zero_hit THEN 1.0 ELSE 0.0 END) / COUNT(*), 4
                )                                    AS zero_hit_rate,
                ROUND(AVG(top1_score), 4)            AS avg_top1_score,
                ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY top1_score), 4) AS p50_score,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY top1_score), 4) AS p95_score,
                ROUND(AVG(latency_ms), 1)            AS avg_latency_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_latency_ms
            FROM base
            GROUP BY report_date, tenant_id, query_modality
        )
        SELECT * FROM agg
        ORDER BY report_date DESC, tenant_id, query_modality
    """).fetchdf()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding Cost KPI
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_kpi(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT
            CAST(e.embed_date AS DATE)               AS report_date,
            e.tenant_id,
            e.embedding_model,
            c.modality,
            COUNT(*)                                 AS embed_request_count,
            SUM(e.prompt_tokens)                     AS total_tokens,
            ROUND(SUM(e.cost_usd), 6)                AS total_cost_usd,
            ROUND(AVG(e.cost_usd), 8)                AS avg_cost_per_chunk
        FROM silver_embeddings e
        JOIN silver_chunks c ON e.chunk_id = c.chunk_id
        GROUP BY
            CAST(e.embed_date AS DATE),
            e.tenant_id,
            e.embedding_model,
            c.modality
        ORDER BY report_date DESC, total_cost_usd DESC
    """).fetchdf()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_gold_layer(current_doc_hashes: dict[str, str]) -> dict[str, pd.DataFrame]:
    """
    Chạy toàn bộ Gold build:
    1. Lấy active Parquet files từ Silver Delta tables
    2. Tạo DuckDB views
    3. Build 3 KPI marts
    4. Ghi ra Gold (Parquet, Delta wrapper)
    """
    from lake_ops import get_active_files, write_table

    chunks_files    = get_active_files(config.SILVER_CHUNKS)
    embed_files     = get_active_files(config.SILVER_EMBEDDINGS)
    retrieval_files = get_active_files(config.SILVER_RETRIEVAL)

    conn = _get_conn(chunks_files, embed_files, retrieval_files)

    freshness = build_freshness_kpi(conn, current_doc_hashes)
    quality   = build_quality_kpi(conn)
    cost      = build_cost_kpi(conn)

    # Ghi Gold tables (Delta)
    os.makedirs(config.GOLD_PATH, exist_ok=True)
    write_table(config.GOLD_FRESHNESS, pa.Table.from_pandas(freshness), mode="overwrite")
    write_table(config.GOLD_QUALITY,   pa.Table.from_pandas(quality),   mode="overwrite")
    write_table(config.GOLD_COST,      pa.Table.from_pandas(cost),      mode="overwrite")

    return {
        "freshness": freshness,
        "quality":   quality,
        "cost":      cost,
    }