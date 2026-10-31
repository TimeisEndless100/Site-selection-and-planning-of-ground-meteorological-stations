# -*- coding: utf-8 -*-
"""GBDT tabular 基线：同信息量下的标准机器学习对照（CPU，分钟级）。

动机（专家评审 P0）：顶级期刊审稿人必问"为什么深度 CNN 胜过梯度提升类 tabular
模型？"——本项目原实验体系无此对照。本模块用**与 PURN-CNN 同源的信息**构建标准
监督 GBDT 基线：

  · 特征 = patches_info.csv 的物理量 slope_max / slope_mean / dem_mean / water_ratio
           + physics RN 标志 is_rn（与 PURN-CNN 的 RN 惩罚同源，同一默认阈值 45°/0.5）
  · 标签 = is_P（全监督，P=1 / 其余=0；与 abla_objective 的 Supervised 配置同为
           "有监督"对照，但架构是标准树模型而非 CNN）
  · 划分 = 与主方法完全相同的空间 5 折（src.spatial_split.make_spatial_folds）
  · 指标 = 与 CNN 相同的 auc / ap / brier / recall_k（同一评估口径）

零 GPU、单机 CPU 分钟级，一次 5 折全跑完。定位：主对比表（组① CNN）的
非 PU 监督基线行，回答"深度学习 + PU 相对标准 tabular 基线是否带来判别增益"。

用法：
    python -m runner.train.gbdt_baseline \
        --processed_dir data/processed --out_dir <dir> --seed 42

输出 <out_dir>/results.csv（折级行，列与 train_cnn 一致，供 analysis.aggregate 消费）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.core.metrics import expected_calibration_error
from src.core.rn import assign_sets
from src.spatial_split import make_spatial_folds

# 与主方法一致的特征/阈值（base.yaml 默认）
FEATURE_COLS = ["slope_max", "slope_mean", "dem_mean", "water_ratio", "is_rn"]
RN_SLOPE_DEFAULT = 45.0
RN_WATER_DEFAULT = 0.5


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, df_val: pd.DataFrame) -> dict:
    """与 train_cnn 完全相同的指标口径。"""
    mask_p = (df_val["set"] == "P").values
    mask_rn = (df_val["set"] == "RN").values
    mask_pu = (df_val["set"] != "RN").values
    score_p_mean = y_score[mask_p].mean() if mask_p.any() else np.nan
    score_rn_mean = y_score[mask_rn].mean() if mask_rn.any() else np.nan
    k = int(mask_p.sum())
    if k > 0:
        top_k = np.argsort(y_score)[::-1][:k]
        recall_at_k = y_true[top_k].sum() / k
    else:
        recall_at_k = 0.0
    # 候选域 P∪U AUC（剔除 RN）：跨模型公平比较主口径。GBDT 用 P-vs-RN 监督训练，
    # 在 P∪U 上评估测试其推广到决策区（候选站址）的能力；RN 属标签来源的"易"负样本。
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
    parser = argparse.ArgumentParser(description="GBDT tabular 基线（CPU，分钟级）")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="outputs/gbdt_baseline")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rn_slope_thresh", type=float, default=RN_SLOPE_DEFAULT)
    parser.add_argument("--rn_water_thresh", type=float, default=RN_WATER_DEFAULT)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--fold_ids", type=str, default="", help="逗号分隔的 fold 子集（诊断用）")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    df = pd.read_csv(processed / "patches_info.csv")
    # is_rn 由 assign_sets 派生（下方 df["is_rn"]），不要求存在于 patches_info.csv
    missing = [c for c in ["slope_max", "slope_mean", "dem_mean", "water_ratio", "is_P", "pid", "lon", "lat"] if c not in df.columns]
    if missing:
        raise SystemExit(f"patches_info.csv 缺列: {missing}")

    # is_rn = physics 规则（与 src/core/rn.assign_sets 严格一致），作为显式特征注入
    df = assign_sets(
        df,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy="physics",
        seed=args.seed,
    )
    df["is_rn"] = (df["set"] == "RN").astype(float)

    folds = make_spatial_folds(df, n_splits=args.folds, grid_size=args.grid_size, seed=args.seed)
    run_fold_ids = set(range(len(folds)))
    if args.fold_ids.strip():
        run_fold_ids = {int(x) for x in args.fold_ids.split(",") if x.strip()}

    X = df[FEATURE_COLS].astype(float).values
    y = df["is_P"].values
    results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx not in run_fold_ids:
            continue
        t0 = time.time()
        # 特征标准化对树模型无必要，直接训练
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=args.seed,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_score = clf.predict_proba(X[val_idx])[:, 1]
        df_val = df.iloc[val_idx]
        m = compute_metrics(y[val_idx], y_score, df_val)
        m.update({"fold": fold_idx, "runtime_min": (time.time() - t0) / 60.0})
        results.append(m)
        print(f"  Fold {fold_idx}: AUC={m['auc']:.4f} | AUC(P∪U)={m['auc_pu']:.4f} | AP={m['ap']:.4f} | "
              f"Brier={m['brier']:.4f} | Recall@{int((df_val['set']=='P').sum())}={m['recall_k']:.4f} | "
              f"{m['runtime_min']:.1f}min")

    if not results:
        raise SystemExit("无折运行（fold_ids 过滤后为空？）")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)
    print(f"GBDT 基线折级结果 → {out_dir / 'results.csv'}")
    print(f"  均值: AUC={results_df['auc'].mean():.4f} | AP={results_df['ap'].mean():.4f} | "
          f"Brier={results_df['brier'].mean():.4f} | Recall@K={results_df['recall_k'].mean():.4f}")


if __name__ == "__main__":
    main()
