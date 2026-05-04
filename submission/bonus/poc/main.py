#!/usr/bin/env python3
"""
main.py — Multimodal RAG Lakehouse PoC
VinUni AI20k | Bonus Challenge Day 18 Track 2

Chạy:  python main.py
Reset: python main.py --reset
"""

import os
import sys
import time
import shutil
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Local modules ────────────────────────────────────────────────────────────
import config
import data_generator as gen
import lake_ops as lake
import gold_layer

# ─── ANSI colors ─────────────────────────────────────────────────────────────
GOLD   = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(text: str, color: str = GOLD) -> None:
    width = 70
    print(f"\n{color}{BOLD}{'─' * width}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'─' * width}{RESET}")


def step(n: int, text: str) -> None:
    print(f"\n{CYAN}{BOLD}[STEP {n}]{RESET} {text}")


def ok(text: str) -> None:
    print(f"  {GREEN}✓{RESET} {text}")


def info(text: str) -> None:
    print(f"  {BLUE}→{RESET} {text}")


def warn(text: str) -> None:
    print(f"  {GOLD}⚠{RESET} {text}")


def table_print(df: pd.DataFrame, max_rows: int = 8) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  {DIM}... ({len(df) - max_rows} rows truncated){RESET}")


def timing(start: float) -> str:
    return f"{DIM}({time.time() - start:.2f}s){RESET}"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Xóa toàn bộ lakehouse trước khi chạy")
    args = parser.parse_args()

    if args.reset and os.path.exists(config.BASE_PATH):
        shutil.rmtree(config.BASE_PATH)
        print(f"{RED}  Đã xóa {config.BASE_PATH}/{RESET}")

    banner("Multimodal RAG Lakehouse — Proof of Concept")
    print(f"  {DIM}Chạy lúc: {datetime.utcnow().isoformat(timespec='seconds')} UTC{RESET}")
    print(f"  {DIM}Config  : {config.N_DOCS} docs | {config.N_DOCS * config.N_CHUNKS_PER_DOC} chunks "
          f"| {config.VECTOR_DIM}-dim vectors | {config.N_RETRIEVAL_EVENTS} retrieval events{RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Generate fake source data
    # ─────────────────────────────────────────────────────────────────────────
    step(1, "Generate fake source data")
    t = time.time()

    docs_arrow   = gen.gen_raw_documents()
    docs_df      = docs_arrow.to_pandas()
    ok(f"Documents  : {len(docs_df):,} rows | modalities: {docs_df['modality'].value_counts().to_dict()}")

    doc_ids      = docs_df["doc_id"].tolist()
    ret_arrow    = gen.gen_raw_retrieval_logs(doc_ids)
    ret_df       = ret_arrow.to_pandas()
    ok(f"Retrieval logs: {len(ret_df):,} rows")
    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Bronze Layer Ingest
    # ─────────────────────────────────────────────────────────────────────────
    step(2, "Bronze Layer — Raw Ingest (Delta tables)")
    t = time.time()

    v_b_docs = lake.write_table(
        config.BRONZE_DOCS, docs_arrow,
        mode="overwrite", partition_by=["ingest_date"]
    )
    ok(f"bronze.raw_documents      → version {v_b_docs} | {len(docs_df):,} rows | "
       f"{len(lake.get_active_files(config.BRONZE_DOCS))} Parquet files")

    v_b_ret = lake.write_table(
        config.BRONZE_RETRIEVAL, ret_arrow,
        mode="overwrite", partition_by=["ingest_date"]
    )
    ok(f"bronze.raw_retrieval_logs → version {v_b_ret} | {len(ret_df):,} rows | "
       f"{len(lake.get_active_files(config.BRONZE_RETRIEVAL))} Parquet files")

    info(timing(t))
    info("Bronze rule: nhận tất cả, không filter, raw_payload là JSON string nguyên gốc")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Silver Layer — Chunking + Mock Embedding
    # ─────────────────────────────────────────────────────────────────────────
    step(3, "Silver Layer — Chunking + Mock Embedding (Delta ACID)")
    t = time.time()

    silver_run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    info(f"silver_run_id = {silver_run_id}")

    # 3a. Document chunks
    chunks_arrow = gen.gen_document_chunks(docs_df, silver_run_id)
    chunks_df    = chunks_arrow.to_pandas()
    v_chunks = lake.write_table(
        config.SILVER_CHUNKS, chunks_arrow,
        mode="overwrite", partition_by=["chunk_date"]
    )
    ok(f"silver.document_chunks    → version {v_chunks} | {len(chunks_df):,} rows")

    # 3b. Embeddings
    info(f"Generating mock embeddings ({config.VECTOR_DIM}-dim, L2-normalized)...")
    embed_arrow = gen.gen_embeddings(chunks_df)
    embed_df    = embed_arrow.to_pandas()
    v_embed = lake.write_table(
        config.SILVER_EMBEDDINGS, embed_arrow,
        mode="overwrite", partition_by=["embed_date"]
    )
    ok(f"silver.embeddings         → version {v_embed} | {len(embed_df):,} rows | "
       f"eval cols = NULL (chưa có feedback)")

    # 3c. Retrieval events
    chunk_ids = chunks_df["chunk_id"].tolist()
    ret_events_arrow = gen.gen_retrieval_events(ret_df, chunk_ids)
    ret_events_df    = ret_events_arrow.to_pandas()
    v_ret_ev = lake.write_table(
        config.SILVER_RETRIEVAL, ret_events_arrow,
        mode="overwrite", partition_by=["event_date"]
    )
    ok(f"silver.retrieval_events   → version {v_ret_ev} | {len(ret_events_df):,} rows | "
       f"user_id và query_text đã hash SHA-256 (PII compliance)")
    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: MERGE Pattern — Re-embed stale documents
    # ─────────────────────────────────────────────────────────────────────────
    step(4, "MERGE Pattern — Simulate document update → re-embed")
    t = time.time()

    # 4a. Simulate 20 documents bị update (doc_hash thay đổi)
    stale_docs_df = gen.gen_stale_documents(docs_df, n_stale=20)
    stale_doc_ids = set(stale_docs_df["doc_id"].tolist())
    info(f"Simulated {len(stale_doc_ids)} documents bị update (doc_hash mới ≠ doc_version_hash trong Silver)")

    stale_chunks_df = chunks_df[chunks_df["doc_id"].isin(stale_doc_ids)].copy()
    info(f"Stale chunks cần re-embed: {len(stale_chunks_df):,}")

    # 4b. Ghi nhớ version TRƯỚC khi MERGE (để time travel sau)
    version_before_merge = lake.get_version(config.SILVER_EMBEDDINGS)
    info(f"silver.embeddings version TRƯỚC MERGE: {version_before_merge}")

    # 4c. Tạo embedding mới cho stale chunks
    reembed_arrow = gen.gen_reembed_batch(stale_chunks_df)
    merge_result  = lake.merge_embeddings(config.SILVER_EMBEDDINGS, reembed_arrow)
    version_after_merge = merge_result["version_after"]

    ok(f"MERGE hoàn thành: version {version_before_merge} → {version_after_merge}")
    ok(f"  {len(stale_chunks_df):,} vectors được update với embedding mới")
    info(f"  _delta_log ghi thêm 1 commit → time travel vẫn xem được vector cũ")
    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Time Travel Demo
    # ─────────────────────────────────────────────────────────────────────────
    step(5, "Time Travel — So sánh vector_norm trước và sau MERGE")
    t = time.time()

    diff_df = lake.time_travel_diff(
        config.SILVER_EMBEDDINGS,
        version_before=version_before_merge,
        version_after=version_after_merge,
        key_col="chunk_id",
        compare_col="vector_norm",
    )

    ok(f"Số chunks có vector_norm thay đổi: {len(diff_df)}")
    if not diff_df.empty:
        print(f"\n  {GOLD}{'chunk_id':36}  {'norm_before':>12}  {'norm_after':>12}  {'delta':>10}{RESET}")
        for _, row in diff_df.head(5).iterrows():
            print(f"  {DIM}{row['chunk_id'][:36]}{RESET}  "
                  f"{row['vector_norm_before']:>12.6f}  "
                  f"{row['vector_norm_after']:>12.6f}  "
                  f"{CYAN}{row['delta']:>+10.6f}{RESET}")
        if len(diff_df) > 5:
            print(f"  {DIM}... ({len(diff_df) - 5} rows more){RESET}")

    # Demo: đọc version cũ
    old_embed_count = len(lake.read_table(config.SILVER_EMBEDDINGS,
                                          version=version_before_merge))
    new_embed_count = len(lake.read_table(config.SILVER_EMBEDDINGS,
                                          version=version_after_merge))
    ok(f"SELECT COUNT(*) VERSION AS OF {version_before_merge} → {old_embed_count:,} rows")
    ok(f"SELECT COUNT(*) VERSION AS OF {version_after_merge}  → {new_embed_count:,} rows")
    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: MERGE Eval Scores (từ retrieval analytics)
    # ─────────────────────────────────────────────────────────────────────────
    step(6, "MERGE Eval Scores — user feedback + LLM-as-judge pipeline")
    t = time.time()

    eval_arrow  = gen.gen_eval_updates(embed_df, ret_events_df)
    eval_result = lake.merge_eval_scores(config.SILVER_EMBEDDINGS, eval_arrow)

    # Kiểm tra eval đã được ghi
    updated_embed = lake.read_table(config.SILVER_EMBEDDINGS)
    eval_filled   = updated_embed["retrieval_hit_count"].notna().sum()
    eval_null     = updated_embed["retrieval_hit_count"].isna().sum()

    ok(f"Eval MERGE: version {eval_result['version_before']} → {eval_result['version_after']}")
    ok(f"Embeddings có eval scores: {eval_filled:,} | Chưa có (chưa được retrieve): {eval_null:,}")
    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Gold Layer — KPI Marts
    # ─────────────────────────────────────────────────────────────────────────
    step(7, "Gold Layer — Build 3 KPI Marts (DuckDB)")
    t = time.time()

    # Build current_doc_hashes: 20 docs có hash mới, còn lại giữ hash cũ
    current_hashes = dict(zip(docs_df["doc_id"], docs_df["doc_hash"]))
    for _, row in stale_docs_df.iterrows():
        current_hashes[row["doc_id"]] = row["doc_hash"]  # hash mới sau update

    gold = gold_layer.build_gold_layer(current_hashes)

    # Print freshness
    ok(f"gold.embedding_freshness_kpi — {len(gold['freshness'])} rows")
    table_print(gold["freshness"][[
        "tenant_id", "modality", "total_chunk_count",
        "stale_chunk_count", "freshness_ratio", "pending_reembed_cost_usd"
    ]])

    print()
    ok(f"gold.retrieval_quality_kpi — {len(gold['quality'])} rows")
    table_print(gold["quality"][[
        "tenant_id", "query_modality", "total_queries",
        "zero_hit_rate", "avg_top1_score", "p95_latency_ms"
    ]])

    print()
    ok(f"gold.embedding_cost_kpi — {len(gold['cost'])} rows")
    table_print(gold["cost"][[
        "tenant_id", "modality", "embed_request_count",
        "total_tokens", "total_cost_usd"
    ]])

    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Stale Detection Demo
    # ─────────────────────────────────────────────────────────────────────────
    step(8, "Stale Detection Pipeline (hourly job)")
    t = time.time()

    freshness_df = gold["freshness"]
    stale_rows   = freshness_df[freshness_df["stale_chunk_count"] > 0]

    total_stale_chunks = int(stale_rows["stale_chunk_count"].sum())
    total_chunks       = int(freshness_df["total_chunk_count"].sum())
    overall_freshness  = round(1 - total_stale_chunks / total_chunks, 4)
    est_reembed_cost   = round(float(freshness_df["pending_reembed_cost_usd"].sum()), 6)

    if total_stale_chunks > 0:
        warn(f"ALERT: {total_stale_chunks:,} / {total_chunks:,} chunks stale "
             f"({(1 - overall_freshness)*100:.1f}%) — freshness_ratio = {overall_freshness}")
        warn(f"       Estimated re-embed cost: ${est_reembed_cost:.6f}")
        for _, row in stale_rows.iterrows():
            warn(f"       [{row['tenant_id']}] {row['modality']:20} "
                 f"→ {int(row['stale_chunk_count'])} stale / {int(row['total_chunk_count'])} total")
    else:
        ok("Tất cả embeddings fresh — không cần re-embed")

    info(timing(t))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Failure Scenario Demos
    # ─────────────────────────────────────────────────────────────────────────
    step(9, "Failure Scenario Demos")

    # Scenario 1: Rollback embedding bug
    print(f"\n  {RED}Scenario 1: Embedding bug → re-embed với model lỗi → cần rollback{RESET}")
    t = time.time()
    current_ver = lake.get_version(config.SILVER_EMBEDDINGS)

    # Inject "buggy" embeddings (tất cả vectors = 0)
    # Lấy đủ tất cả columns mà MERGE cần update
    bug_rows = embed_df.head(50)[[
        "embedding_id", "chunk_id", "doc_id", "tenant_id",
        "embedding_model", "embed_ts", "embed_date",
        "prompt_tokens", "cost_usd", "chunk_silver_run_id",
        "retrieval_hit_count", "avg_relevance_score", "eval_updated_ts",
    ]].copy()
    bug_rows["vector_norm"] = 0.0   # bug: norm = 0 → useless vector
    import pyarrow as pa
    bug_schema = pa.schema([
        pa.field("embedding_id",         pa.string()),
        pa.field("chunk_id",             pa.string()),
        pa.field("doc_id",              pa.string()),
        pa.field("tenant_id",           pa.string()),
        pa.field("embedding_model",     pa.string()),
        pa.field("embed_ts",            pa.timestamp("us")),
        pa.field("embed_date",          pa.date32()),
        pa.field("prompt_tokens",       pa.int32()),
        pa.field("cost_usd",            pa.float64()),
        pa.field("chunk_silver_run_id", pa.string()),
        pa.field("retrieval_hit_count", pa.int32()),
        pa.field("avg_relevance_score", pa.float64()),
        pa.field("eval_updated_ts",     pa.timestamp("us")),
        pa.field("vector_norm",         pa.float64()),
    ])
    bug_vecs = [[0.0] * config.VECTOR_DIM] * len(bug_rows)
    bug_base = pa.Table.from_pandas(bug_rows, schema=bug_schema, preserve_index=False)
    bug_arrow = bug_base.append_column(
        pa.field("vector", pa.list_(pa.float32())),
        pa.array(bug_vecs, type=pa.list_(pa.float32()))
    )
    lake.merge_embeddings(config.SILVER_EMBEDDINGS, bug_arrow)
    buggy_ver = lake.get_version(config.SILVER_EMBEDDINGS)

    # Verify bug
    buggy_df = lake.read_table(config.SILVER_EMBEDDINGS)
    n_zero   = (buggy_df["vector_norm"] == 0.0).sum()
    warn(f"  Sau bug MERGE: {n_zero} embeddings có vector_norm = 0.0 (version {buggy_ver})")

    # RESTORE
    restore_ver = lake.restore_table(config.SILVER_EMBEDDINGS, current_ver)
    restored_df = lake.read_table(config.SILVER_EMBEDDINGS)
    n_zero_after = (restored_df["vector_norm"] == 0.0).sum()
    ok(f"  RESTORE TO VERSION {current_ver} → current version = {restore_ver}")
    ok(f"  Sau restore: {n_zero_after} embeddings có vector_norm = 0.0 ✓")
    info(timing(t))

    # Scenario 2: Decree-13 right-to-be-forgotten
    print(f"\n  {RED}Scenario 2: Decree-13 — right-to-be-forgotten cho tenant 'acme'{RESET}")
    t = time.time()
    from deltalake import DeltaTable

    before_count = len(lake.read_table(config.SILVER_RETRIEVAL))
    acme_count   = (lake.read_table(config.SILVER_RETRIEVAL)["tenant_id"] == "acme").sum()
    info(f"  silver.retrieval_events: {before_count:,} rows | acme: {acme_count:,} rows")

    # DELETE acme rows
    dt_ret = DeltaTable(config.SILVER_RETRIEVAL)
    dt_ret.delete("tenant_id = 'acme'")
    after_count = len(lake.read_table(config.SILVER_RETRIEVAL))
    ok(f"  DELETE WHERE tenant_id='acme': {before_count:,} → {after_count:,} rows")
    ok(f"  (VACUUM RETAIN 0 HOURS sẽ xóa physical files — bỏ qua PoC để giữ time travel)")
    info(timing(t))

    # Scenario 3: Zero-hit alert
    print(f"\n  {RED}Scenario 3: Retrieval quality drop — CEO alert{RESET}")
    quality_df = gold["quality"]
    high_zero  = quality_df[quality_df["zero_hit_rate"] > 0.25]
    if not high_zero.empty:
        for _, row in high_zero.iterrows():
            warn(f"  ALERT [{row['tenant_id']}] {row['query_modality']:20} "
                 f"zero_hit_rate = {row['zero_hit_rate']:.2%} "
                 f"(ngưỡng 25%) → kiểm tra stale embeddings")
    else:
        ok("  zero_hit_rate < 25% cho tất cả tenants/modalities")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Demo Queries
    # ─────────────────────────────────────────────────────────────────────────
    step(10, "Demo Queries — DuckDB trên Gold tables")

    import duckdb
    conn = duckdb.connect(":memory:")

    def reg_gold(name: str, path: str) -> None:
        files = lake.get_active_files(path)
        if files:
            files_lit = ", ".join(f"'{f}'" for f in files)
            conn.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet([{files_lit}])")

    reg_gold("freshness_kpi",  config.GOLD_FRESHNESS)
    reg_gold("quality_kpi",    config.GOLD_QUALITY)
    reg_gold("cost_kpi",       config.GOLD_COST)

    print(f"\n  {GOLD}Q1: Tenant nào có freshness_ratio thấp nhất?{RESET}")
    q1 = conn.execute("""
        SELECT tenant_id,
               SUM(stale_chunk_count) AS stale,
               SUM(total_chunk_count) AS total,
               ROUND(1.0 - SUM(stale_chunk_count)*1.0/SUM(total_chunk_count), 4) AS freshness
        FROM freshness_kpi
        GROUP BY tenant_id
        ORDER BY freshness ASC
    """).fetchdf()
    table_print(q1)

    print(f"\n  {GOLD}Q2: Chi phí embedding mỗi modality tuần này{RESET}")
    q2 = conn.execute("""
        SELECT modality,
               SUM(embed_request_count) AS requests,
               SUM(total_tokens) AS tokens,
               ROUND(SUM(total_cost_usd), 6) AS cost_usd
        FROM cost_kpi
        GROUP BY modality
        ORDER BY cost_usd DESC
    """).fetchdf()
    table_print(q2)

    print(f"\n  {GOLD}Q3: Modality nào có chất lượng retrieval tệ nhất?{RESET}")
    q3 = conn.execute("""
        SELECT query_modality,
               SUM(total_queries) AS queries,
               ROUND(AVG(zero_hit_rate), 4) AS avg_zero_hit,
               ROUND(AVG(avg_top1_score), 4) AS avg_score,
               ROUND(AVG(p95_latency_ms), 0) AS p95_lat_ms
        FROM quality_kpi
        GROUP BY query_modality
        ORDER BY avg_zero_hit DESC
    """).fetchdf()
    table_print(q3)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11: Delta History
    # ─────────────────────────────────────────────────────────────────────────
    step(11, "Delta History — silver.embeddings transaction log")
    history = lake.get_history(config.SILVER_EMBEDDINGS)
    print(f"\n  {GOLD}{'Version':>7}  {'Operation':25}  {'Timestamp'}{RESET}")
    for h in sorted(history, key=lambda x: x.get("version", 0)):
        ver  = h.get("version", "?")
        op   = h.get("operation", "?")
        ts   = h.get("timestamp", "")
        if isinstance(ts, int):
            ts = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {DIM}{ver:>7}{RESET}  {op:25}  {DIM}{ts}{RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    banner("Summary — Toàn bộ pipeline hoàn thành", GREEN)

    total_rows = (
        len(lake.read_table(config.BRONZE_DOCS)) +
        len(lake.read_table(config.BRONZE_RETRIEVAL)) +
        len(lake.read_table(config.SILVER_CHUNKS)) +
        len(lake.read_table(config.SILVER_EMBEDDINGS)) +
        len(lake.read_table(config.SILVER_RETRIEVAL)) +
        len(lake.read_table(config.GOLD_FRESHNESS)) +
        len(lake.read_table(config.GOLD_QUALITY)) +
        len(lake.read_table(config.GOLD_COST))
    )

    layers = [
        ("Bronze", [
            ("raw_documents",      config.BRONZE_DOCS),
            ("raw_retrieval_logs", config.BRONZE_RETRIEVAL),
        ]),
        ("Silver", [
            ("document_chunks",    config.SILVER_CHUNKS),
            ("embeddings",         config.SILVER_EMBEDDINGS),
            ("retrieval_events",   config.SILVER_RETRIEVAL),
        ]),
        ("Gold", [
            ("embedding_freshness_kpi", config.GOLD_FRESHNESS),
            ("retrieval_quality_kpi",   config.GOLD_QUALITY),
            ("embedding_cost_kpi",      config.GOLD_COST),
        ]),
    ]

    for layer_name, tables in layers:
        print(f"\n  {GOLD}{BOLD}{layer_name} Layer{RESET}")
        for tname, tpath in tables:
            stats = lake.table_stats(tpath)
            print(f"    {GREEN}✓{RESET} {tname:30} "
                  f"v{stats['version']} | "
                  f"{stats['row_count']:>6,} rows | "
                  f"{stats['file_count']} files")

    print(f"\n  {BOLD}Total rows across all tables:{RESET} {total_rows:,}")
    print(f"  {BOLD}Vector dimensions:{RESET} {config.VECTOR_DIM}")
    print(f"  {BOLD}Delta features demonstrated:{RESET}")
    print(f"    {GREEN}✓{RESET} MERGE (re-embed stale documents)")
    print(f"    {GREEN}✓{RESET} Time Travel (VERSION AS OF)")
    print(f"    {GREEN}✓{RESET} RESTORE (rollback embedding bug)")
    print(f"    {GREEN}✓{RESET} DELETE + compliance (Decree-13)")
    print(f"    {GREEN}✓{RESET} Transaction history (_delta_log)")
    print(f"    {GREEN}✓{RESET} Partitioning (by date)")
    print(f"\n  {DIM}Lakehouse data: {config.BASE_PATH}/{RESET}")
    print(f"  {DIM}Để reset: python main.py --reset{RESET}\n")


if __name__ == "__main__":
    main()