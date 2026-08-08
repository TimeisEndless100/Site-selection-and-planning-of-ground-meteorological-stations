# -*- coding: utf-8 -*-
"""实验矩阵（job/result 注册表）的共享操作。

plan 与 aggregate 都要处理同一份 experiment_matrix.csv：按 result_file/command
生成稳定 job_key、合并快照时按 key 覆盖旧行。三处历史重复实现
（run_closure_plan.py / run_closure_aggregate.py）统一收拢到本模块。
"""
from __future__ import annotations

import pandas as pd


def ensure_job_key(df: pd.DataFrame) -> pd.DataFrame:
    """无 job_key 列时为每行生成稳定 key：result_file → command → stage:exp_id:idx。"""
    if "job_key" in df.columns:
        return df.copy()
    out = df.copy()
    keys: list[str] = []
    for idx, row in out.iterrows():
        result_file = str(row.get("result_file", "") or "")
        command = str(row.get("command", "") or "")
        stage = str(row.get("stage", "") or "")
        exp_id = str(row.get("exp_id", "") or "")
        key = result_file or command or f"{stage}:{exp_id}:{idx}"
        keys.append(key)
    out["job_key"] = keys
    return out


def prefer_current_row(existing: dict, current: dict) -> dict:
    """合并冲突时保留「更已完成」的行：非 planned 优先，其次保留已有状态。"""
    current_status = str(current.get("status", "planned") or "planned")
    existing_status = str(existing.get("status", "planned") or "planned")
    if current_status != "planned":
        return current
    if existing_status in {"", "planned", "nan", "None"}:
        return current
    return existing


def merge_matrices(existing_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """按 job_key 合并两个矩阵，已存在的 key 用 prefer_current_row 取舍，保留列并集。"""
    existing_df = ensure_job_key(existing_df)
    current_df = ensure_job_key(current_df)
    all_cols = list(dict.fromkeys(existing_df.columns.tolist() + current_df.columns.tolist()))
    existing_df = existing_df.reindex(columns=all_cols)
    current_df = current_df.reindex(columns=all_cols)

    merged: list[dict] = []
    key_to_idx: dict[str, int] = {}

    for _, row in existing_df.iterrows():
        rec = row.to_dict()
        key = str(rec["job_key"])
        key_to_idx[key] = len(merged)
        merged.append(rec)

    for _, row in current_df.iterrows():
        rec = row.to_dict()
        key = str(rec["job_key"])
        if key in key_to_idx:
            merged[key_to_idx[key]] = prefer_current_row(merged[key_to_idx[key]], rec)
        else:
            key_to_idx[key] = len(merged)
            merged.append(rec)

    return pd.DataFrame(merged, columns=all_cols)
