# -*- coding: utf-8 -*-
"""论文数据图（← scripts/viz_result.py + scripts/viz_heatmap_pixel.py + viz_folds.py）。

三个旧脚本合为一个模块，子命令分发：
    python -m reporting.figures result   --infer_csv … [--out_png …]          # 结果散点图
    python -m reporting.figures heatmap  --infer_csv … [--out_png …]          # 像素级热力图（对比度拉伸）
    python -m reporting.figures probmap  --merged_csv … [--out_png …]         # 概率地图 vs 地形地图 2×2 面板
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


def _pivot_grid(df: pd.DataFrame, value_col: str, grid_max: int = 600) -> tuple[pd.DataFrame, list[float]]:
    """把 lon/lat + value 按目标格网分箱重聚合成 2D 矩阵与 extent。

    注意：patch 坐标为逐格唯一（本数据集 ~22k 个唯一 lat/lon），若直接
    pivot_table(index="lat", columns="lon") 会生成 22175×22175 稠密矩阵
    （~491M 格，~4GB），在低内存机器上极慢/直接内存不足。这里按目标格网
    尺寸（grid_max×grid_max）把每格坐标分箱、取均值，得到有界的光滑概率面
    ——这正是"把预测概率重聚合为地图"的意图。cell 均值为该格内 patch 均值。
    """
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    n_lat = max(1, min(int(grid_max), int(df["lat"].nunique())))
    n_lon = max(1, min(int(grid_max), int(df["lon"].nunique())))
    cell_lat = (lat_max - lat_min) / n_lat
    cell_lon = (lon_max - lon_min) / n_lon
    lat_idx = np.clip(((df["lat"].values - lat_min) / cell_lat).astype(int), 0, n_lat - 1)
    lon_idx = np.clip(((df["lon"].values - lon_min) / cell_lon).astype(int), 0, n_lon - 1)
    tmp = df.assign(_lat=lat_idx, _lon=lon_idx)
    grid = tmp.pivot_table(index="_lat", columns="_lon", values=value_col, aggfunc="mean")
    grid = grid.reindex(index=range(n_lat), columns=range(n_lon))  # 空行/空列补 NaN
    extent = [lon_min, lon_max, lat_min, lat_max]
    return grid, extent


def cmd_probmap(args: argparse.Namespace) -> None:
    """概率地图 vs 地形地图 2×2 面板：消费 analysis/prob_map.py 的 prob_map_merged.csv。

    A 概率地图 / B 坡度 / C 水域比 / D 概率地图 + RN（物理不可行格）掩膜——
    直接展示模型概率面相对地形约束的吻合与差异（差异格 = 高概率但被掩膜的不可行格）。
    """
    df = pd.read_csv(args.merged_csv)
    grid_p, extent = _pivot_grid(df, "prob_mean", args.grid)
    grid_s, _ = _pivot_grid(df, "slope_max", args.grid)
    grid_w, _ = _pivot_grid(df, "water_ratio", args.grid)

    # 概率百分位拉伸（与 heatmap 一致）
    raw = df["prob_mean"].values
    vmin = np.nanpercentile(raw, args.stretch_min)
    vmax = np.nanpercentile(raw, args.stretch_max)

    # D 面板：RN 格掩膜（NaN → 灰色），其余显示概率——差异格 = 高概率但被掩膜的不可行格。
    # 分箱后每格可能混合 RN/非 RN patch，取格内 RN 占比，占比 ≥50% 判定为不可行格。
    rn_frac, _ = _pivot_grid(df, "is_rn", args.grid)
    grid_masked = grid_p.where(rn_frac < 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    panels = [
        (axes[0, 0], grid_p, "RdYlBu_r", vmin, vmax, "A. Model Probability Map"),
        (axes[0, 1], grid_s, "YlOrBr", np.nanpercentile(df["slope_max"], 1), np.nanpercentile(df["slope_max"], 99), "B. Terrain: Slope (max)"),
        (axes[1, 0], grid_w, "Blues", np.nanpercentile(df["water_ratio"], 1), np.nanpercentile(df["water_ratio"], 99), "C. Terrain: Water Ratio"),
        (axes[1, 1], grid_masked, "RdYlBu_r", vmin, vmax, "D. Probability vs Terrain (RN masked)"),
    ]
    for ax, grid, cmap, lo, hi, title in panels:
        im = ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        if title.startswith("A") or title.startswith("D"):
            cbar.set_label(f"P [{lo:.2f},{hi:.2f}]", fontsize=9)
            df_p = df[df["is_P"] == 1]
            ax.scatter(df_p["lon"], df_p["lat"], c="black", s=4, marker="+", linewidth=0.5, alpha=0.7, zorder=3)
        elif title.startswith("B"):
            cbar.set_label("slope_max", fontsize=9)
        else:
            cbar.set_label("water_ratio", fontsize=9)
    fig.suptitle("Suitability Probability Map vs Terrain Map (Chongqing)", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"Prob map comparison saved to {args.out_png}")


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
    parser = argparse.ArgumentParser(description="论文数据图（子命令：result / heatmap / probmap / folds）")
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

    p = sub.add_parser("probmap", help="概率地图 vs 地形地图 2×2 面板（消费 prob_map_merged.csv）")
    p.add_argument("--merged_csv", default="outputs/exp_closure/prob_map_merged.csv", help="analysis/prob_map 输出的合并表")
    p.add_argument("--out_png", default="outputs/prob_map_vs_terrain.png")
    p.add_argument("--stretch_min", type=float, default=2, help="概率对比度拉伸下限百分位 (0-100)")
    p.add_argument("--stretch_max", type=float, default=98, help="概率对比度拉伸上限百分位 (0-100)")
    p.add_argument("--grid", type=int, default=600, help="重聚合格网尺寸（grid×grid；patch 坐标逐格唯一，直接 pivot 会内存爆炸，按此分箱）")
    p.add_argument("--dpi", type=int, default=300, help="输出 PNG 分辨率（出稿用 300；调试可调低加速）")

    p = sub.add_parser("folds", help="空间分折分布图（与训练一致的划分）")
    p.add_argument("--patches_csv", default="data/processed/patches_info.csv")
    p.add_argument("--out_png", default="outputs/spatial_folds_map.png")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    {"result": cmd_result, "heatmap": cmd_heatmap, "probmap": cmd_probmap, "folds": cmd_folds}[args.command](args)


if __name__ == "__main__":
    main()
