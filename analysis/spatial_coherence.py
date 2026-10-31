# -*- coding: utf-8 -*-
"""空间连贯性分析（CPU 秒级，消费 app_infer 三个整域概率场）。

动机（2026-08-12 评审）：GNN 在 AUC(P∪U) 上的增益很小（+0.004 量级），论文主张改为
**空间连贯性**而非"GNN 提高准确率"：GNN 的价值是把 CNN 的逐补丁概率场 refine 成更
空间连贯（平滑、区域一致）的场，且把不确定度集中在复杂地形。本模块给出三条证据：

1. **概率场 Moran's I**（用 GNN 自身图结构 k=graph_k=8 的 kNN 邻接，行标准化）：
   沿 CNN-PN → PURN-CNN → PURN+GNN 链应递增 → 空间自相关/连贯性递增。
   （标签行 is_P 的 Moran's I 作为参照：正 = 真实站址空间聚集。）
2. **残差空间自相关**（候选域 P∪U 上 resid = is_P − prob）：越低越接近随机 →
   模型已吸收空间结构、未留下系统性空间误差。
3. **折叠集成不确定度 vs 地形复杂度**：prob_std（5 折模型整域推理的 std，app_infer 已产出）
   与 slope_max 的相关性 + 复杂地形（RN 格）内/外 std 对比 → 不确定度集中在地形歧义区。

输入推理 CSV 仅需 pid + prob_mean + prob_std（坐标/标签/slope 统一从 patches_info.csv 按 pid 合并）。

用法：
    python -m analysis.spatial_coherence \
        --patches_csv data/processed/patches_info.csv \
        --infer_csvs "CNN-PN:<pn>/inference_results.csv,PURN-CNN:<purn>/inference_results.csv,PURN+GNN:<sage>/inference_results_sage.csv" \
        --graph_k 8 --rn_slope_thresh 45.0 --rn_water_thresh 0.5 \
        --out_dir outputs/exp_rerun_*/app_spatial_coherence

输出 <out_dir>/spatial_coherence_summary.csv（每模型一行）+ <out_dir>/uncertainty_merged.csv。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.core.rn import assign_sets


def _moran_i(x: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """行标准化 kNN 邻接的 Moran's I。

    正值 = 空间聚集（同质块），负值 = 空间分散，~0 = 空间随机。
    用 kNN（k 与 GNN 自身 graph_k 一致）而非距离阈值：局部邻域尺度远小于
    区域范围，lon/lat 未投影的畸变可忽略；行标准化使权重矩阵每行和为 1。
    """
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)) or len(x) < 2:
        return float("nan")
    xm = x - np.nanmean(x)
    denom = float(np.nansum(xm**2))
    if denom == 0.0:
        return float("nan")
    tree = cKDTree(np.asarray(coords, dtype=float))
    # k+1：首位是自身（距离 0）
    _, idx = tree.query(coords, k=min(k + 1, len(coords)), workers=-1)
    neighbors = np.asarray(idx)
    if neighbors.ndim == 1:  # k+1 >= n 的退化情形
        neighbors = np.broadcast_to(neighbors[:, None], (len(coords), min(k + 1, len(coords))))
    self_mask = neighbors[:, 0] == np.arange(len(coords)) if neighbors.shape[1] > 1 else np.ones(len(coords), bool)
    nbr = neighbors[:, 1:]
    num = float(np.nansum(xm[:, None] * xm[nbr])) / k
    return num / denom


def _load_infer(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} 缺列: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="空间连贯性分析（CPU，消费 app_infer 整域概率场）")
    parser.add_argument("--patches_csv", type=str, default="data/processed/patches_info.csv")
    parser.add_argument("--infer_csvs", type=str, required=True,
                        help="label:path 逗号分隔（如 CNN-PN:<pn>/inference_results.csv,PURN-CNN:...,PURN+GNN:...）")
    parser.add_argument("--graph_k", type=int, default=8, help="kNN 邻接 k（与 GNN 自身 graph_k 一致）")
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0, help="RN 物理规则斜率阈值（与主流程 assign_sets 一致）")
    parser.add_argument("--rn_water_thresh", type=float, default=0.5, help="RN 物理规则水域阈值")
    parser.add_argument("--out_dir", type=str, default="outputs/app_spatial_coherence")
    args = parser.parse_args()

    patches = pd.read_csv(args.patches_csv)
    # 注意：patches_info.csv 的 set 列可能是旧预处理遗留（无 RN）；RN 由物理规则派生
    missing = [c for c in ["pid", "lon", "lat", "is_P", "slope_max", "water_ratio"] if c not in patches.columns]
    if missing:
        raise SystemExit(f"patches_info.csv 缺列: {missing}")
    patches = assign_sets(patches, rn_slope_thresh=args.rn_slope_thresh,
                          rn_water_thresh=args.rn_water_thresh, rn_strategy="physics", seed=42)
    patches = patches.set_index("pid")

    # 标签 Moran's I（真实站址空间聚集程度，作为解释参照）
    coords = patches[["lon", "lat"]].values
    label_moran = _moran_i(patches["is_P"].values.astype(float), coords, k=args.graph_k)

    pairs = [s.strip() for s in args.infer_csvs.split(",") if s.strip()]
    rows: list[dict] = []
    merged = patches.copy()

    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"infer_csvs 项需 'label:path'：{pair}")
        label, path = pair.split(":", 1)
        path = path.strip()
        if not path:
            continue
        df = _load_infer(Path(path), ["pid", "prob_mean", "prob_std"])
        # 推理 CSV 可能自带 lon/lat/set 等与 patches 重叠列 → 只保留需合并的三列
        df = df[["pid", "prob_mean", "prob_std"]].set_index("pid")
        d = patches.join(df, how="inner")
        if len(d) < 2:
            print(f"  [skip] {label}: pid 对齐后无数据")
            continue
        prob = d["prob_mean"].values.astype(float)
        std = d["prob_std"].values.astype(float)
        coords_d = d[["lon", "lat"]].values  # 与 d 同行序对齐（inner join 保左序）

        moran_prob = _moran_i(prob, coords_d, k=args.graph_k)
        # 候选域残差（is_P − prob）空间自相关：越低越随机
        pu = (d["set"] != "RN").values
        resid = d["is_P"].values.astype(float) - prob
        if pu.sum() > 1:
            moran_resid = _moran_i(resid[pu], coords_d[pu], k=args.graph_k)
        else:
            moran_resid = float("nan")
        # 不确定度 vs 地形复杂度
        corr_std_slope = float(np.corrcoef(std, d["slope_max"].values.astype(float))[0, 1])
        rn_terr = (d["set"] == "RN").values
        std_rn = float(std[rn_terr].mean()) if rn_terr.any() else float("nan")
        std_non_rn = float(std[~rn_terr].mean()) if (~rn_terr).any() else float("nan")

        rows.append({
            "model": label,
            "n_patches": int(len(d)),
            "moran_prob": round(moran_prob, 4),
            "moran_resid_pu": round(moran_resid, 4),
            "corr_std_slope": round(corr_std_slope, 4),
            "std_mean": round(float(std.mean()), 4),
            "std_rn_terrain": round(std_rn, 4),
            "std_non_rn": round(std_non_rn, 4),
            "prob_mean": round(float(prob.mean()), 4),
        })
        merged[f"{label}_prob"] = pd.Series(prob, index=d.index)
        merged[f"{label}_std"] = pd.Series(std, index=d.index)
        print(f"  {label}: Moran(prob)={moran_prob:.4f} | Moran(resid P∪U)={moran_resid:.4f} "
              f"| corr(std,slope)={corr_std_slope:.4f} | std_RN={std_rn:.4f} std_非RN={std_non_rn:.4f}")

    # 标签行作参照（无模型残差）
    rows.insert(0, {
        "model": "Labels(is_P)",
        "n_patches": len(patches),
        "moran_prob": round(label_moran, 4),
        "moran_resid_pu": float("nan"),
        "corr_std_slope": float("nan"),
        "std_mean": float("nan"),
        "std_rn_terrain": float("nan"),
        "std_non_rn": float("nan"),
        "prob_mean": float("nan"),
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "spatial_coherence_summary.csv", index=False)
    merged.reset_index().to_csv(out_dir / "uncertainty_merged.csv", index=False)
    print(f"空间连贯性汇总 → {out_dir / 'spatial_coherence_summary.csv'}")
    print(summary[["model", "moran_prob", "moran_resid_pu", "corr_std_slope", "std_rn_terrain", "std_non_rn"]].to_string(index=False))


if __name__ == "__main__":
    main()
