# -*- coding: utf-8 -*-
"""论文数据图（← scripts/viz_result.py + scripts/viz_heatmap_pixel.py + viz_folds.py）。

三个旧脚本合为一个模块，子命令分发：
    python -m reporting.figures result   --infer_csv … [--out_png …]          # 结果散点图
    python -m reporting.figures heatmap  --infer_csv … [--out_png …]          # 像素级热力图（对比度拉伸）
    python -m reporting.figures probmap  --merged_csv … [--out_png …]         # 概率地图 vs 地形地图 2×2 面板
    python -m reporting.figures folds    [--patches_csv …] [--out_png …]      # 空间折划分分布图

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
from matplotlib.colors import BoundaryNorm


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


def _grid_interp(
    df: pd.DataFrame,
    grid_size: int,
    value_col: str = "prob_mean",
) -> tuple[np.ndarray, list[float]]:
    """把离散点概率插值为规则格网连续光滑概率面（grid_size×grid_size）。

    与 _pivot_grid（分箱取均值，6.2% 格点有值 → 大片 NaN 空白）不同：
    用 scipy.interpolate.griddata 从全部原始点做 linear 插值，天然填满整个
    矩形 extent（无空白）；插值后 gaussian_filter 平滑过渡色带。返回 (网格, extent)。
    内存有界（grid_size² 的 float64），替代会 OOM 的 pivot_table。
    """
    from scipy import interpolate, ndimage

    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    xs = np.linspace(lon_min, lon_max, grid_size)
    ys = np.linspace(lat_min, lat_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    pts = df[["lon", "lat"]].values
    vals = df[value_col].values.astype(float)

    grid = interpolate.griddata(pts, vals, (xx, yy), method="linear")
    # linear 在数据凸包外产生 NaN → 用 nearest 兜底填满
    if np.isnan(grid).any():
        nearest = interpolate.griddata(pts, vals, (xx, yy), method="nearest")
        grid = np.where(np.isnan(grid), nearest, grid)
    # 高斯平滑消除插值网格的锯齿过渡（sigma 单位 = 格点数）
    grid = ndimage.gaussian_filter(grid, sigma=1.0)

    extent = [lon_min, lon_max, lat_min, lat_max]
    return grid, extent


def _mask_boundary(grid: np.ndarray, extent: list[float], boundary_file: str) -> np.ndarray:
    """边界 polygon 掩膜：把边界外的格点置 NaN（不画到矩形四角）。

    用 shapely.prepared 加速逐点 within 判定（600×600 = 36 万点，纯 Python
    Point().within 会明显慢）；union_all() 兼容新 geopandas（unary_union 已废弃）。
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.prepared import prep

    gdf = gpd.read_file(boundary_file)
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    poly = gdf.geometry.union_all()
    prepared = prep(poly)

    n_lat, n_lon = grid.shape
    xs = np.linspace(extent[0], extent[1], n_lon)
    ys = np.linspace(extent[2], extent[3], n_lat)
    xx, yy = np.meshgrid(xs, ys)
    inside = np.array([prepared.contains(Point(x, y)) for x, y in zip(xx.ravel(), yy.ravel())]).reshape(n_lat, n_lon)
    return np.where(inside, grid, np.nan)


def _lattice_pivot(df: pd.DataFrame, value_col: str = "prob_mean") -> tuple[np.ndarray, list[float]]:
    """把散点 patch 还原为原生栅格矩阵（复现用户 pivot 逻辑 + 坐标吸附修正）。

    直接 pivot_table(index='lat', columns='lon') 在本数据集不可行：22175 个 patch 的
    lon/lat 是全精度浮点、逐单元唯一（实测 unique_lon=unique_lat=22175），pivot 生成
    22175×22175 ≈ 3.9GB 对角稀疏矩阵，非 NaN 格互不相邻 → 渲染为散布孤立像素。

    但 patch 实际落在 ~1.9km 规则格网上（pid 形如 p_{row}_{col}，每 64 个原始像元
    取一个；重庆边界内格网覆盖率实测 96.6%）。先把坐标吸附回格网：
      row_bin = row // 64, col_bin = col // 64，
    每 row_bin 取中位 lat、每 col_bin 取中位 lon，再 pivot_table，
    得 230×240 原生分辨率矩阵，重庆界内无缝、无插值、无性能问题。
    注意栅格相对经纬轴旋转 ~1.7°，轴对齐近似会把边界侧错位 ~6–13 km（可后续加仿射变换修正）。

    返回 (矩阵, extent)。extent 按用户脚本含半像元偏移。
    """
    df = df[df["pid"].str.startswith("p_")].copy()
    parts = df["pid"].str.split("_", expand=True)
    rb = parts[1].astype(int).values // 64
    cb = parts[2].astype(int).values // 64
    snap = df.copy()
    snap["rb"] = rb
    snap["cb"] = cb
    snap["lat_snap"] = snap.groupby("rb")["lat"].transform("mean")
    snap["lon_snap"] = snap.groupby("cb")["lon"].transform("mean")
    m = snap.pivot_table(index="lat_snap", columns="lon_snap", values=value_col)
    lat_vals = m.index.values.astype(float)
    lon_vals = m.columns.values.astype(float)
    res_x = (lon_vals.max() - lon_vals.min()) / (len(lon_vals) - 1)
    res_y = (lat_vals.max() - lat_vals.min()) / (len(lat_vals) - 1)
    extent = [lon_vals.min() - res_x / 2, lon_vals.max() + res_x / 2,
              lat_vals.min() - res_y / 2, lat_vals.max() + res_y / 2]
    return m.values, extent


def cmd_heatmap(args: argparse.Namespace) -> None:
    """像素级栅格热力图：pivot 还原 2D 矩阵（默认）或插值平滑 + 百分位对比度拉伸 + 边界/站点叠加。

    复现用户脚本逻辑（2026-08-15）：
    - 核心用 pivot_table 把离散点还原为 2D 矩阵（原生格网分辨率，interpolation='nearest' 像素感）。
      唯一必要修正：patch 坐标逐单元唯一，需先吸附回 ~1.9km 规则格网（见 _lattice_pivot），
      否则 pivot 生成 22175×22175 对角稀疏矩阵 → 大片空白。
    - 对比度拉伸：百分位（--stretch_min/max，默认忽略最低/最高 2%）或 Min-Max（--use_minmax）。
    - extent 含半像元偏移；叠加 geojson 边界 + 现有站点（黑色 +）；colorbar 标注拉伸区间。
    - 备选 --method interp：griddata linear 插值 + gaussian 平滑（填满全部格点，色带连续）。
    """
    import os
    import geopandas as gpd

    df = pd.read_csv(args.infer_csv)

    if args.method == "pivot":
        grid_matrix, extent = _lattice_pivot(df)
        interpolation = "nearest"
        print(f"Rasterized via lattice pivot: {grid_matrix.shape[0]}x{grid_matrix.shape[1]} "
              f"(native ~1.9km grid, coverage {np.isfinite(grid_matrix).mean() * 100:.1f}%)")
    else:
        grid_matrix, extent = _grid_interp(df, args.grid)
        interpolation = "bilinear"
        print(f"Rasterized via griddata interp: {grid_matrix.shape[0]}x{grid_matrix.shape[1]}")

    # 可选：边界 polygon 掩膜，边界外不画
    if args.mask_boundary and args.boundary_file and os.path.exists(args.boundary_file):
        grid_matrix = _mask_boundary(grid_matrix, extent, args.boundary_file)
        print("Boundary masking applied (cells outside Chongqing → NaN)")

    raw_values = df["prob_mean"].values
    if args.use_minmax:
        vmin, vmax = raw_values.min(), raw_values.max()
        print(f"Contrast Strategy: Min-Max [{vmin:.4f}, {vmax:.4f}]")
    else:
        vmin = np.nanpercentile(raw_values, args.stretch_min)
        vmax = np.nanpercentile(raw_values, args.stretch_max)
        print(f"Contrast Strategy: Percentile ({args.stretch_min}-{args.stretch_max}%) [{vmin:.4f}, {vmax:.4f}]")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(grid_matrix, extent=extent, origin="lower", cmap=args.cmap,
                   vmin=vmin, vmax=vmax, interpolation=interpolation, zorder=1)

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

    注意：patch 坐标为逐单元唯一（本数据集 ~22k 个唯一 lat/lon），若直接
    pivot_table(index="lat", columns="lon") 会生成 22175×22175 稠密矩阵
    （~491M 单元，~4GB），在低内存机器上极慢/直接内存不足。这里按目标格网
    尺寸（grid_max×grid_max）把每单元坐标分箱、取均值，得到有界的光滑概率面
    ——这正是"把预测概率重聚合为地图"的意图。cell 均值为该单元内 patch 均值。
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

    A 概率地图 / B 坡度 / C 水域比 / D 概率地图 + RN（物理不可行单元）掩膜——
    直接展示模型概率面相对地形约束的吻合与差异（差异单元 = 高概率但被掩膜的不可行单元）。
    """
    df = pd.read_csv(args.merged_csv)
    grid_p, extent = _pivot_grid(df, "prob_mean", args.grid)
    grid_s, _ = _pivot_grid(df, "slope_max", args.grid)
    grid_w, _ = _pivot_grid(df, "water_ratio", args.grid)

    # 概率百分位拉伸（与 heatmap 一致）
    raw = df["prob_mean"].values
    vmin = np.nanpercentile(raw, args.stretch_min)
    vmax = np.nanpercentile(raw, args.stretch_max)

    # D 面板：RN 单元掩膜（NaN → 灰色），其余显示概率——差异单元 = 高概率但被掩膜的不可行单元。
    # 分箱后每单元可能混合 RN/非 RN patch，取单元内 RN 占比，占比 ≥50% 判定为不可行单元。
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


def _grid_blocks_mode(df: pd.DataFrame, grid_size: int, value_col: str = "prob_mean") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """按训练一致的栅格块方式把 lon/lat 分成 grid_size×grid_size 块，块值 = 众数档位。

    与 src/spatial_split._prepare_spatial_blocks 相同的 pd.cut 分箱（lon_bin/lat_bin），
    返回：网格矩阵（值 = 块内 value_col 的众数档位，无数据块为 NaN）、块内样本数矩阵、extent。

    为什么用众数而非均值：patch 级四分位断点间距很小（Q1≈0.062 / Q2≈0.070 / Q3≈0.077），
    块内均值会把概率压窄到中位数附近，导致高/低档位在块级消失（实测 10×10 下 Q4 一档
    0 块）。先给每 patch 标档位再取块内众数，块颜色 = 块内多数 patch 所在的档位，
    四分位差异在块级可读。与训练折划分的栅格块一致，消除 pivot 大面积留白。
    """
    lon_bin = pd.cut(df["lon"], bins=grid_size, labels=False)
    lat_bin = pd.cut(df["lat"], bins=grid_size, labels=False)
    g = df.assign(_lon=lon_bin, _lat=lat_bin)
    agg = (
        g.groupby(["_lat", "_lon"], sort=True)
        .agg(value=(value_col, lambda s: s.mode(dropna=True).iloc[0] if len(s) else np.nan),
             n=(value_col, "size"))
        .reset_index()
    )
    grid = np.full((grid_size, grid_size), np.nan)
    count = np.zeros((grid_size, grid_size), dtype=int)
    for _, r in agg.iterrows():
        grid[int(r["_lat"]), int(r["_lon"])] = r["value"]
        count[int(r["_lat"]), int(r["_lon"])] = int(r["n"])
    extent = [float(df["lon"].min()), float(df["lon"].max()),
              float(df["lat"].min()), float(df["lat"].max())]
    return grid, count, extent


def cmd_quartile(args: argparse.Namespace) -> None:
    """概率地图：按训练栅格块（grid_size×grid_size）重构 + 四分位等频分箱。

    把整域概率按 25/50/75 百分位切成 4 个等频档位，每个栅格块取块内概率均值
    映射到所属档位，按离散色阶渲染——与连续色标不同，直接呈现「概率高/低
    区域在哪里、各占多少」，且与训练折划分的栅格块一致，无大面积留白。
    消费 prob_map_merged.csv（推理概率 + lon/lat/is_P + 地形）。
    """
    df = pd.read_csv(args.merged_csv)
    probs = df["prob_mean"].values

    # 四分位断点（等频：每档 25% 单元）
    q1, q2, q3 = np.nanpercentile(probs, [25, 50, 75])
    labels = ["Q1", "Q2", "Q3", "Q4"]

    # 逐单元档位（用于统计 + out_csv）
    df = df.assign(
        quartile=pd.cut(df["prob_mean"], bins=[-np.inf, q1, q2, q3, np.inf], labels=labels, right=True)
    )

    # 各档统计
    counts = df["quartile"].value_counts()
    means = df.groupby("quartile", observed=True)["prob_mean"].mean()
    rng = df.groupby("quartile", observed=True)["prob_mean"].agg(lambda s: (s.min(), s.max()))
    print("四分位档位统计（patch 级等频）：")
    for lab in labels:
        n = int(counts.get(lab, 0))
        m = means.get(lab, np.nan)
        r = rng.get(lab, (np.nan, np.nan))
        print(f"  {lab}: n={n:5d} ({n / len(df) * 100:5.1f}%)  mean={m:.4f}  range=[{r[0]:.4f}, {r[1]:.4f}]")

    # 先给每 patch 标档位码（1–4），再按训练栅格块取块内众数档位
    df = df.assign(q_code=df["quartile"].cat.codes + 1)  # Q1→1 … Q4→4
    grid_q, count, extent = _grid_blocks_mode(df, args.grid_size, value_col="q_code")
    mask = ~np.isnan(grid_q)

    # 块级档位占比（供读图）
    if count.sum() > 0:
        for q in range(1, 5):
            n_q = int((grid_q[mask] == q).sum())
            n_data = int(mask.sum())
            print(f"  块级档位 Q{q}: {n_q:3d} 块 ({n_q / n_data * 100:4.1f}%)")
    print(f"  栅格块 {args.grid_size}×{args.grid_size}，有数据块 {int(mask.sum())}/{args.grid_size * args.grid_size}，"
          f"平均每块 {count[mask].mean():.0f} patch")

    fig, ax = plt.subplots(figsize=(10, 9), dpi=args.dpi)
    cmap = plt.get_cmap(getattr(args, "cmap", "YlOrRd")).copy()  # Q1 低概率→浅黄，Q4 高概率→深红
    cmap.set_bad("#e8e8e8")  # 无数据块（重庆边界外）浅灰，与留白区分
    norm = BoundaryNorm([1, 2, 3, 4], cmap.N)
    im = ax.imshow(grid_q, extent=extent, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, ticks=[1.5, 2.5, 3.5, 4.5])
    q_bounds = [f"[{probs.min():.3f},{q1:.3f}]", f"[{q1:.3f},{q2:.3f}]",
                f"[{q2:.3f},{q3:.3f}]", f"[{q3:.3f},{probs.max():.3f}]"]
    cbar.ax.set_yticklabels([f"{lab}\n{q_b}" for lab, q_b in zip(labels, q_bounds)])
    cbar.set_label("Prediction Probability (quartile)", fontsize=10)

    # 栅格块网格线：标出「按训练栅格块划分」的块结构
    lon_edges = np.linspace(extent[0], extent[1], args.grid_size + 1)
    lat_edges = np.linspace(extent[2], extent[3], args.grid_size + 1)
    for xe in lon_edges:
        ax.axvline(xe, color="white", lw=0.5, alpha=0.6, zorder=2)
    for ye in lat_edges:
        ax.axhline(ye, color="white", lw=0.5, alpha=0.6, zorder=2)

    # 叠加现有站点
    df_p = df[df["is_P"] == 1]
    ax.scatter(df_p["lon"], df_p["lat"], c="white", s=6, marker="+", linewidth=0.6,
               alpha=0.85, zorder=3, label="Existing stations")

    ax.set_title(f"Meteorological Station Buildability Map (Quartile-binned, {args.grid_size}x{args.grid_size} Grid)",
                 fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"Quartile probability map saved to {args.out_png}")

    if args.out_csv:
        out = df[["pid", "lon", "lat", "prob_mean", "is_P", "quartile"]].copy()
        out.to_csv(args.out_csv, index=False)
        print(f"Quartile-labelled rows saved to {args.out_csv}")


def cmd_folds(args: argparse.Namespace) -> None:
    """空间折划分分布图：与训练一致的 make_spatial_folds 划分着色（← 顶层 viz_folds.py）。"""
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
    parser = argparse.ArgumentParser(description="论文数据图（子命令：result / heatmap / probmap / quartile / folds）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("result", help="结果散点图（所有 U 点概率着色 + 现有站点）")
    p.add_argument("--infer_csv", required=True)
    p.add_argument("--out_png", default="result_map.png")

    p = sub.add_parser("heatmap", help="像素级栅格热力图（对比度拉伸 + 边界/站点叠加）")
    p.add_argument("--infer_csv", required=True, help="推理生成的 CSV 文件")
    p.add_argument("--boundary_file", default="data/raw/重庆市_市.geojson", help="边界文件")
    p.add_argument("--out_png", default="suitability_heatmap_pixel.png")
    p.add_argument("--title", default="Meteorological Station Suitability (Contrast Enhanced)")
    p.add_argument("--stretch_min", type=float, default=2, help="对比度拉伸下限百分位 (0-100)")
    p.add_argument("--stretch_max", type=float, default=98, help="对比度拉伸上限百分位 (0-100)")
    p.add_argument("--use_minmax", action="store_true", help="强制使用 Min-Max 归一化 (0-1) 而非百分位拉伸")
    p.add_argument("--method", choices=["pivot", "interp"], default="pivot",
                   help="栅格化方法：pivot=吸附到原生 ~1.9km 格网后 pivot_table（像素级无缝，默认）；interp=griddata 插值平滑")
    p.add_argument("--grid", type=int, default=600, help="--method interp 时的插值格网尺寸（grid×grid）")
    p.add_argument("--cmap", default="YlOrRd", help="暖色调顺序色标（默认 YlOrRd 暖黄→红；可选 viridis/OrRd/hot 等）")
    p.add_argument("--mask_boundary", action="store_true", help="用边界 polygon 掩膜，边界外的插值结果置 NaN（只显示研究区内）")

    p = sub.add_parser("probmap", help="概率地图 vs 地形地图 2×2 面板（消费 prob_map_merged.csv）")
    p.add_argument("--merged_csv", default="outputs/exp_closure/prob_map_merged.csv", help="analysis/prob_map 输出的合并表")
    p.add_argument("--out_png", default="outputs/prob_map_vs_terrain.png")
    p.add_argument("--stretch_min", type=float, default=2, help="概率对比度拉伸下限百分位 (0-100)")
    p.add_argument("--stretch_max", type=float, default=98, help="概率对比度拉伸上限百分位 (0-100)")
    p.add_argument("--grid", type=int, default=600, help="重聚合格网尺寸（grid×grid；patch 坐标逐单元唯一，直接 pivot 会内存爆炸，按此分箱）")
    p.add_argument("--dpi", type=int, default=300, help="输出 PNG 分辨率（出稿用 300；调试可调低加速）")

    p = sub.add_parser("quartile", help="概率地图：按训练栅格块重构 + 四分位等频分箱（消费 prob_map_merged.csv）")
    p.add_argument("--merged_csv", default="outputs/exp_closure/prob_map_merged.csv", help="analysis/prob_map 输出的合并表")
    p.add_argument("--out_png", default="outputs/prob_map_quartile.png")
    p.add_argument("--out_csv", default="", help="可选：带四分位标签的行写出（pid/lon/lat/prob_mean/is_P/quartile）")
    p.add_argument("--grid_size", type=int, default=10, help="栅格块尺寸（与训练折划分一致，默认 10×10）")
    p.add_argument("--cmap", default="YlOrRd", help="色标（默认暖色 YlOrRd；viridis 冷色可选）")
    p.add_argument("--dpi", type=int, default=300, help="输出 PNG 分辨率")

    p = sub.add_parser("folds", help="空间折划分分布图（与训练一致的划分）")
    p.add_argument("--patches_csv", default="data/processed/patches_info.csv")
    p.add_argument("--out_png", default="outputs/spatial_folds_map.png")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    {"result": cmd_result, "heatmap": cmd_heatmap, "probmap": cmd_probmap, "quartile": cmd_quartile, "folds": cmd_folds}[args.command](args)


if __name__ == "__main__":
    main()
