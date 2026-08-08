# -*- coding: utf-8 -*-
"""论文数据图（← scripts/viz_result.py + scripts/viz_heatmap_pixel.py + viz_folds.py）。

三个旧脚本合为一个模块，子命令分发：
    python -m reporting.figures result   --infer_csv … [--out_png …]          # 结果散点图
    python -m reporting.figures heatmap  --infer_csv … [--out_png …]          # 像素级热力图（对比度拉伸）
    python -m reporting.figures folds    [--patches_csv …] [--out_png …]      # 空间分折分布图

只消费 analysis/runner 的输出 CSV，不自己算统计（reporting 层职责边界）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cmd_result(args: argparse.Namespace) -> None:
    """结果地图：所有 U 点用概率着色，叠加现有站点（← viz_result.py）。"""
    df = pd.read_csv(args.infer_csv)
    plt.figure(figsize=(10, 8))
    plt.scatter(df["lon"], df["lat"], c=df["prob_mean"], cmap="RdYlBu_r", s=1, alpha=0.5, label="Suitability")
    plt.colorbar(label="Buildability Probability")
    existing = df[df["is_P"] == 1]
    plt.scatter(existing["lon"], existing["lat"], c="black", s=5, marker="x", alpha=0.5, label="Existing")
    plt.title("Meteorological Station Buildability Map (AI Predicted)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.savefig(args.out_png, dpi=300)
    print(f"Map saved to {args.out_png}")


def cmd_heatmap(args: argparse.Namespace) -> None:
    """像素级栅格热力图：pivot 还原 2D 矩阵 + 百分位对比度拉伸（← viz_heatmap_pixel.py）。"""
    import os
    import geopandas as gpd

    df = pd.read_csv(args.infer_csv)
    grid_matrix = df.pivot_table(index="lat", columns="lon", values="prob_mean")

    lons = grid_matrix.columns.values
    lats = grid_matrix.index.values
    res_x = (lons.max() - lons.min()) / (len(lons) - 1)
    res_y = (lats.max() - lats.min()) / (len(lats) - 1)
    extent = [lons.min() - res_x / 2, lons.max() + res_x / 2, lats.min() - res_y / 2, lats.max() + res_y / 2]

    raw_values = df["prob_mean"].values
    if args.use_minmax:
        vmin, vmax = raw_values.min(), raw_values.max()
        print(f"Contrast Strategy: Min-Max [{vmin:.4f}, {vmax:.4f}]")
    else:
        vmin = np.nanpercentile(raw_values, args.stretch_min)
        vmax = np.nanpercentile(raw_values, args.stretch_max)
        print(f"Contrast Strategy: Percentile ({args.stretch_min}-{args.stretch_max}%) [{vmin:.4f}, {vmax:.4f}]")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(grid_matrix, extent=extent, origin="lower", cmap="RdYlBu_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest", zorder=1)

    if args.boundary_file and os.path.exists(args.boundary_file):
        try:
            gdf_bound = gpd.read_file(args.boundary_file)
            if gdf_bound.crs and gdf_bound.crs.to_string() != "EPSG:4326":
                gdf_bound = gdf_bound.to_crs("EPSG:4326")
            gdf_bound.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)
        except Exception as e:
            print(f"Warning: Could not load boundary: {e}")

    df_p = df[df["is_P"] == 1]
    ax.scatter(df_p["lon"], df_p["lat"], c="black", s=5, marker="+", linewidth=0.5, alpha=0.6,
               label="Existing Stations", zorder=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(f"Suitability Probability (Stretched [{vmin:.2f}, {vmax:.2f}])", fontsize=10)
    ax.set_title(args.title, fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print(f"Map saved to {args.out_png}")


def cmd_folds(args: argparse.Namespace) -> None:
    """空间分折分布图：与训练一致的 make_spatial_folds 划分着色（← 顶层 viz_folds.py）。"""
    from src.spatial_split import make_spatial_folds

    df = pd.read_csv(args.patches_csv)
    folds = make_spatial_folds(df, n_splits=args.n_splits, grid_size=args.grid_size, seed=args.seed)

    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(folds):
        df.loc[val_idx, "fold"] = fold_idx

    plt.figure(figsize=(10, 8), dpi=300)
    cmap = plt.get_cmap("Set1", args.n_splits)
    sc = plt.scatter(df["lon"], df["lat"], c=df["fold"], cmap=cmap, s=2, alpha=0.8, marker="s")
    cbar = plt.colorbar(sc, ticks=range(args.n_splits), fraction=0.03, pad=0.04)
    cbar.set_label("Fold ID (Validation Set)")
    plt.title(f"Spatial Block Cross-Validation ({args.grid_size}x{args.grid_size} Grid)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print(f"Fold map saved to {args.out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="论文数据图（子命令：result / heatmap / folds）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("result", help="结果散点图（所有 U 点概率着色 + 现有站点）")
    p.add_argument("--infer_csv", required=True)
    p.add_argument("--out_png", default="result_map.png")

    p = sub.add_parser("heatmap", help="像素级栅格热力图（对比度拉伸）")
    p.add_argument("--infer_csv", required=True, help="推理生成的 CSV 文件")
    p.add_argument("--boundary_file", default="data/raw/重庆市_市.geojson", help="边界文件")
    p.add_argument("--out_png", default="suitability_heatmap_pixel.png")
    p.add_argument("--title", default="Meteorological Station Suitability (Contrast Enhanced)")
    p.add_argument("--stretch_min", type=float, default=2, help="对比度拉伸下限百分位 (0-100)")
    p.add_argument("--stretch_max", type=float, default=98, help="对比度拉伸上限百分位 (0-100)")
    p.add_argument("--use_minmax", action="store_true", help="强制使用 Min-Max 归一化 (0-1) 而非百分位拉伸")

    p = sub.add_parser("folds", help="空间分折分布图（与训练一致的划分）")
    p.add_argument("--patches_csv", default="data/processed/patches_info.csv")
    p.add_argument("--out_png", default="outputs/spatial_folds_map.png")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    {"result": cmd_result, "heatmap": cmd_heatmap, "folds": cmd_folds}[args.command](args)


if __name__ == "__main__":
    main()
