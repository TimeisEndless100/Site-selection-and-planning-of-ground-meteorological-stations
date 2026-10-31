# -*- coding: utf-8 -*-
"""概率地图 vs 地形地图差异分析。

将模型整域推理的预测概率"重聚合为地图"，用物理地形约束（slope/water/DEM）做空间校验：
- **Spearman 相关**：prob_mean ↔ 各地形特征（学得概率是否与地形规则方向一致）。
- **RN 吻合**：物理不可行单元（RN 判定，复用 src/core/rn.assign_sets）vs 非 RN 单元的
  prob_mean 均值差——模型是否给物理不可行单元低概率。
- **概率十分位 → 地形均值**：高概率单元是否集中在地形可行区。
- **高概率单元（top 10%）物理可行率**：学得模型是否尊重物理约束。

消费 app_infer 的推理输出（默认最终模型 purn_gnn 的 inference_results_sage.csv）
与 patches_info.csv（地形特征）。管线作业由 runner.plan 的 render_app_prob_map 生成。

用法：
    python -m analysis.prob_map --infer_csv <infer.csv> --patches_csv data/processed/patches_info.csv --out_dir <dir>

输出（均落 out_dir）：
- prob_map_summary.csv   长表（metric, feature, value, n）：相关 + RN 吻合 + top10 可行率
- prob_map_deciles.csv   概率十分位 → 各地形均值（宽表）
- prob_map_merged.csv    逐单元合并表（prob + terrain + is_rn + prob_decile），供 reporting.figures probmap 渲染
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.core.rn import assign_sets

# 参与相关分析的地形特征（patches_info.csv 列）
TERRAIN_FEATURES = ["slope_max", "slope_mean", "water_ratio", "dem_mean"]


def load_merged(infer_csv: Path, patches_csv: Path) -> pd.DataFrame:
    """推理输出 × patches 按 pid 合并，产出含 prob + 地形特征的逐单元表。"""
    df_infer = pd.read_csv(infer_csv)
    df_patch = pd.read_csv(patches_csv)

    required = ["pid", "prob_mean"]
    missing = [c for c in required if c not in df_infer.columns]
    if missing:
        raise ValueError(f"infer_csv 缺列: {missing}（需含 {required}）")
    missing_p = [c for c in ["pid", *TERRAIN_FEATURES] if c not in df_patch.columns]
    if missing_p:
        raise ValueError(f"patches_csv 缺列: {missing_p}")

    df = df_infer[["pid", "lon", "lat", "prob_mean"]].copy()
    if "prob_std" in df_infer.columns:
        df["prob_std"] = df_infer["prob_std"].astype(float)
    else:
        df["prob_std"] = np.nan

    df = df.merge(
        df_patch[["pid", *TERRAIN_FEATURES, "is_P"]],
        on="pid",
        how="inner",
    )
    if len(df) == 0:
        raise ValueError("infer 与 patches 按 pid 合并后为空（pid 未对齐？）")
    return df


def summarize(df: pd.DataFrame, rn_slope_thresh: float, rn_water_thresh: float) -> list[dict]:
    """计算相关 / RN 吻合 / top10 可行率，返回长表行（metric, feature, value, n）。"""
    rows: list[dict] = []

    # 1. Spearman 相关
    for feat in TERRAIN_FEATURES:
        valid = df[["prob_mean", feat]].dropna()
        if len(valid) < 10:
            rows.append({"metric": "spearman_rho", "feature": feat, "value": np.nan, "n": 0})
            rows.append({"metric": "spearman_p", "feature": feat, "value": np.nan, "n": 0})
            continue
        rho, p = spearmanr(valid["prob_mean"], valid[feat])
        rows.append({"metric": "spearman_rho", "feature": feat, "value": float(rho), "n": len(valid)})
        rows.append({"metric": "spearman_p", "feature": feat, "value": float(p), "n": len(valid)})

    # 2. RN 吻合（物理不可行单元 vs 非 RN 单元）
    df_rn = assign_sets(
        df,
        rn_slope_thresh=rn_slope_thresh,
        rn_water_thresh=rn_water_thresh,
        rn_strategy="physics",
    )
    is_rn = df_rn["set"].values == "RN"
    df = df.copy()
    df["is_rn"] = is_rn

    prob_rn = df.loc[is_rn, "prob_mean"]
    prob_non = df.loc[~is_rn, "prob_mean"]
    rows.append({"metric": "prob_mean", "feature": "rn_cell", "value": float(prob_rn.mean()) if len(prob_rn) else np.nan, "n": int(is_rn.sum())})
    rows.append({"metric": "prob_mean", "feature": "non_rn_cell", "value": float(prob_non.mean()) if len(prob_non) else np.nan, "n": int((~is_rn).sum())})
    rows.append({"metric": "prob_diff_non_rn_minus_rn", "feature": "all", "value": float(prob_non.mean() - prob_rn.mean()) if len(prob_rn) and len(prob_non) else np.nan, "n": int(len(df))})
    rows.append({"metric": "rn_frac", "feature": "all", "value": float(is_rn.mean()), "n": int(len(df))})

    # 3. 高概率单元（top 10%）物理可行率
    k = max(1, int(0.1 * len(df)))
    top_idx = np.argsort(df["prob_mean"].values)[::-1][:k]
    top = df.iloc[top_idx]
    rows.append({"metric": "feasible_frac", "feature": "top10", "value": float((~top["is_rn"]).mean()), "n": k})
    rows.append({"metric": "prob_mean", "feature": "top10", "value": float(top["prob_mean"].mean()), "n": k})
    for feat in TERRAIN_FEATURES:
        rows.append({"metric": f"mean_{feat}", "feature": "top10", "value": float(top[feat].mean()), "n": k})

    # 供后续复用的 is_rn 落回 df（deciles / merged 需要）
    df["is_rn"] = is_rn
    return rows, df


def assign_deciles(df: pd.DataFrame, n_deciles: int) -> pd.DataFrame:
    """给逐单元表加 prob_decile 列（pd.qcut 十分位；重复边界自动合并）。"""
    df = df.copy()
    df["prob_decile"] = pd.qcut(
        df["prob_mean"], q=n_deciles, labels=False, retbins=False, duplicates="drop"
    )
    return df


def decile_table(df: pd.DataFrame) -> pd.DataFrame:
    """概率十分位 → 地形均值 + 可行率（宽表，供应用表/图）。"""
    if "prob_decile" not in df.columns:
        raise ValueError("decile_table 需先 assign_deciles")
    rows = []
    for d, g in df.groupby("prob_decile"):
        rows.append(
            {
                "decile": int(d),
                "n": int(len(g)),
                "prob_min": float(g["prob_mean"].min()),
                "prob_max": float(g["prob_mean"].max()),
                "mean_slope_max": float(g["slope_max"].mean()),
                "mean_slope_mean": float(g["slope_mean"].mean()),
                "mean_water_ratio": float(g["water_ratio"].mean()),
                "mean_dem_mean": float(g["dem_mean"].mean()),
                "feasible_rate": float((~g["is_rn"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="概率地图 vs 地形地图差异分析")
    parser.add_argument("--infer_csv", required=True, help="推理输出（app_infer，默认最终模型 purn_gnn）")
    parser.add_argument("--patches_csv", default="data/processed/patches_info.csv")
    parser.add_argument("--out_dir", default="outputs/exp_closure")
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument("--n_deciles", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged(Path(args.infer_csv), Path(args.patches_csv))
    print(f"Loaded {len(df)} merged patches (infer={Path(args.infer_csv).name})")

    summary_rows, df = summarize(df, args.rn_slope_thresh, args.rn_water_thresh)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "prob_map_summary.csv"
    summary.to_csv(summary_path, index=False)

    df = assign_deciles(df, args.n_deciles)
    deciles = decile_table(df)
    deciles_path = out_dir / "prob_map_deciles.csv"
    deciles.to_csv(deciles_path, index=False)

    merged_path = out_dir / "prob_map_merged.csv"
    merged_cols = ["pid", "lon", "lat", "prob_mean", "prob_std", "is_P", *TERRAIN_FEATURES, "is_rn", "prob_decile"]
    df[merged_cols].to_csv(merged_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {deciles_path}")
    print(f"Saved: {merged_path}")


if __name__ == "__main__":
    main()
