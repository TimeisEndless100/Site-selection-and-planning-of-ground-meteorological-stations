# -*- coding: utf-8 -*-
"""physics-only 参考基线：无学习、纯物理规则的朴素分类器（CPU 秒级）。

动机（专家评审 P0）：RN 规则（slope_max>45° 或 water_ratio>0.5）吃掉 ~60% 域，
审稿人会问"判别力有多少来自物理先验本身，多少来自学习？"。本模块给出
**无任何学习**的参照行：

    y_score = 1.0 if (not is_rn) else 0.0   # 物理可建 → 站址正样本；物理不可行 → 0

与主方法同数据、同空间 5 折、同指标口径。**双口径解读（重要，避免循环论证）**：

- `auc`（全域，P vs RN+U）：≈ 0.81，是**由标签构成决定的常数**而非分类技能——P 全部非 RN
  （RN 规则正是产生标签的规则），故全域 AUC 恒等于 (n_RN + 0.5·n_U)/n_nonP。
  此数不可与 CNN 全域 valAUC 直接比较（口径不可比，且物理规则"看着答案打分"）。
- `auc_pu`（候选域 P∪U，剔除 RN）：恒 = 0.5。候选站址全部打 1.0 → 完全无判别。
  这恰是论文论点：**物理规则只能筛掉不可建地，无法在候选站址中排序，判别力全部
  来自学习**。CNN/GBDT 在 P∪U 上的 auc_pu 才是公平可比的学习增量。

确定性结果，一次运行全场一致。

用法：
    python -m analysis.physics_baseline \
        --processed_dir data/processed --out_dir <dir> --seed 42

输出 <out_dir>/results.csv（折级行，列与 train_cnn 一致，供 analysis.aggregate 消费，
stage=cmp_pu_baseline, model=Physics-only）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.core.metrics import expected_calibration_error
from src.core.rn import assign_sets
from src.spatial_split import make_spatial_folds


def physics_metrics(y_true: np.ndarray, y_score: np.ndarray, df_val: pd.DataFrame) -> dict:
    """与 train_cnn / gbdt_baseline 完全相同的指标口径。

    P∪U 上物理打分恒为 1.0（全并列）→ auc_pu = 0.5，正是"规则无候选域判别力"的证明。
    """
    mask_p = (df_val["set"] == "P").values
    mask_rn = (df_val["set"] == "RN").values
    mask_pu = (df_val["set"] != "RN").values
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    score_p_mean = y_score[mask_p].mean() if mask_p.any() else np.nan
    score_rn_mean = y_score[mask_rn].mean() if mask_rn.any() else np.nan
    k = int(mask_p.sum())
    if k > 0:
        top_k = np.argsort(y_score)[::-1][:k]
        recall_at_k = y_true[top_k].sum() / k
    else:
        recall_at_k = 0.0
    auc_pu = (
        float(roc_auc_score(y_true[mask_pu], y_score[mask_pu]))
        if mask_pu.sum() > 1 and len(np.unique(y_true[mask_pu])) > 1
        else float("nan")
    )
    ece = expected_calibration_error(y_true, y_score)
    ece_pu = (
        expected_calibration_error(y_true[mask_pu], y_score[mask_pu])
        if mask_pu.sum() > 1 and len(np.unique(y_true[mask_pu])) > 1
        else float("nan")
    )
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "auc_pu": auc_pu,
        "ap": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": ece,
        "ece_pu": ece_pu,
        "separation": float(score_p_mean - score_rn_mean) if not np.isnan(score_p_mean) and not np.isnan(score_rn_mean) else np.nan,
        "recall_k": float(recall_at_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="physics-only 无学习参考基线（CPU 秒级）")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="outputs/physics_baseline")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    df = pd.read_csv(processed / "patches_info.csv")
    missing = [c for c in ["is_P", "lon", "lat", "slope_max", "water_ratio"] if c not in df.columns]
    if missing:
        raise SystemExit(f"patches_info.csv 缺列: {missing}")

    df = assign_sets(
        df,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy="physics",
        seed=args.seed,
    )
    y = df["is_P"].values
    # 无学习：物理可建 → 站址正样本（1.0）；物理不可行 → 0.0
    y_score_all = (df["set"] != "RN").astype(float).values

    folds = make_spatial_folds(df, n_splits=args.folds, grid_size=args.grid_size, seed=args.seed)
    results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        df_val = df.iloc[val_idx]
        m = physics_metrics(y[val_idx], y_score_all[val_idx], df_val)
        m.update({"fold": fold_idx, "runtime_min": 0.0})
        results.append(m)
        print(f"  Fold {fold_idx}: AUC={m['auc']:.4f} | AUC(P∪U)={m['auc_pu']:.4f} | AP={m['ap']:.4f} | Brier={m['brier']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)
    print(f"physics-only 折级结果 → {out_dir / 'results.csv'}")
    print(f"  均值: AUC={results_df['auc'].mean():.4f} | AUC(P∪U)={results_df['auc_pu'].mean():.4f} | AP={results_df['ap'].mean():.4f}")
    print(f"  注: 全域 AUC 是 RN 占比决定的常数；AUC(P∪U)=0.5 证明物理规则无候选域判别力")


if __name__ == "__main__":
    main()
