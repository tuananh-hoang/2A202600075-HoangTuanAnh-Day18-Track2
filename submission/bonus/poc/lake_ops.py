"""
lake_ops.py — Delta Lake read/write/merge utilities

Wrapper mỏng quanh deltalake Python API (delta-rs 1.x).
Tất cả logic Delta tập trung ở đây để main.py chỉ gọi high-level functions.
"""

import os
import shutil
from typing import Optional

import pyarrow as pa
import pandas as pd
from deltalake import DeltaTable, write_deltalake


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_table(path: str, table: pa.Table,
                mode: str = "overwrite",
                partition_by: Optional[list[str]] = None) -> int:
    """
    Ghi PyArrow Table vào Delta path.
    Trả về version sau khi ghi.
    """
    os.makedirs(path, exist_ok=True)
    kwargs = {"mode": mode}
    if partition_by:
        kwargs["partition_by"] = partition_by

    write_deltalake(path, table, **kwargs)
    return DeltaTable(path).version()


def append_table(path: str, table: pa.Table,
                 partition_by: Optional[list[str]] = None) -> int:
    return write_table(path, table, mode="append", partition_by=partition_by)


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_table(path: str, version: Optional[int] = None) -> pd.DataFrame:
    """Đọc Delta table (hoặc 1 version cụ thể cho time travel)."""
    if version is not None:
        dt = DeltaTable(path, version=version)
    else:
        dt = DeltaTable(path)
    return dt.to_pandas()


def get_version(path: str) -> int:
    return DeltaTable(path).version()


def get_history(path: str) -> list[dict]:
    return DeltaTable(path).history()


def get_active_files(path: str) -> list[str]:
    """Trả về danh sách Parquet files active trong Delta table hiện tại."""
    return DeltaTable(path).file_uris()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_embeddings(table_path: str, source: pa.Table) -> dict:
    """
    MERGE source vào silver.embeddings.
    - MATCHED: update vector, embed_ts, cost_usd, vector_norm, embedding_model
    - NOT MATCHED: insert new row

    Trả về dict với thông tin số rows affected và version mới.
    """
    dt = DeltaTable(table_path)
    version_before = dt.version()

    result = (
        dt.merge(
            source=source,
            predicate="target.chunk_id = source.chunk_id",
            source_alias="source",
            target_alias="target",
        )
        .when_matched_update({
            "embedding_id":    "source.embedding_id",
            "vector":          "source.vector",
            "vector_norm":     "source.vector_norm",
            "embedding_model": "source.embedding_model",
            "embed_ts":        "source.embed_ts",
            "embed_date":      "source.embed_date",
            "prompt_tokens":   "source.prompt_tokens",
            "cost_usd":        "source.cost_usd",
        })
        .when_not_matched_insert_all()
        .execute()
    )

    version_after = DeltaTable(table_path).version()
    return {
        "version_before": version_before,
        "version_after":  version_after,
        "metrics":        result,
    }


def merge_eval_scores(table_path: str, source: pa.Table) -> dict:
    """
    MERGE eval scores (hit_count, avg_relevance_score) vào Silver embeddings.
    Chỉ MATCHED UPDATE — không insert row mới.
    """
    dt = DeltaTable(table_path)
    version_before = dt.version()

    result = (
        dt.merge(
            source=source,
            predicate="target.chunk_id = source.chunk_id",
            source_alias="source",
            target_alias="target",
        )
        .when_matched_update({
            "retrieval_hit_count":  "source.retrieval_hit_count",
            "avg_relevance_score":  "source.avg_relevance_score",
            "eval_updated_ts":      "source.eval_updated_ts",
        })
        .execute()
    )

    version_after = DeltaTable(table_path).version()
    return {
        "version_before": version_before,
        "version_after":  version_after,
        "metrics":        result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Time Travel
# ─────────────────────────────────────────────────────────────────────────────

def time_travel_diff(path: str,
                     version_before: int,
                     version_after: int,
                     key_col: str = "chunk_id",
                     compare_col: str = "vector_norm") -> pd.DataFrame:
    """
    So sánh giá trị của compare_col giữa 2 versions.
    Trả về DataFrame với các rows bị thay đổi.
    """
    df_before = read_table(path, version=version_before)
    df_after  = read_table(path, version=version_after)

    merged = df_before[[key_col, compare_col]].merge(
        df_after[[key_col, compare_col]],
        on=key_col,
        suffixes=("_before", "_after"),
    )
    changed = merged[
        merged[f"{compare_col}_before"] != merged[f"{compare_col}_after"]
    ].copy()
    changed["delta"] = (
        changed[f"{compare_col}_after"] - changed[f"{compare_col}_before"]
    )
    return changed


def restore_table(path: str, target_version: int) -> int:
    """
    RESTORE TABLE về target_version.
    Trong delta-rs 1.x, restore bằng cách ghi lại snapshot của version cũ.
    Trả về version mới sau restore.
    """
    dt = DeltaTable(path)
    dt.restore(target_version)
    return DeltaTable(path).version()


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def table_stats(path: str) -> dict:
    dt = DeltaTable(path)
    df = dt.to_pandas()
    return {
        "version":    dt.version(),
        "row_count":  len(df),
        "file_count": len(dt.file_uris()),
        "columns":    list(df.columns),
    }


def drop_table(path: str) -> None:
    """Xóa hoàn toàn Delta table (dùng cho reset PoC)."""
    if os.path.exists(path):
        shutil.rmtree(path)