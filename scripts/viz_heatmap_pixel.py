# -*- coding: utf-8 -*-
"""
脚本：绘制高对比度、像素级栅格热力图
改进点：
1. 使用 pivot 将离散点还原为 2D 矩阵，实现像素级无缝覆盖。
2. 引入对比度拉伸 (Percentile Stretching)，解决概率分布集中导致对比度低的问题。
"""
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_csv", required=True, help="推理生成的 CSV 文件")
    parser.add_argument("--boundary_file", default="data/raw/重庆市_市.geojson", help="边界文件")
    parser.add_argument("--out_png", default="suitability_heatmap_pixel.png")
    parser.add_argument("--title", default="Meteorological Station Suitability (Contrast Enhanced)")

    # 对比度调整参数
    parser.add_argument("--stretch_min", type=float, default=2,
                        help="对比度拉伸下限百分位 (0-100)，例如 2 表示忽略最低 2% 的值")
    parser.add_argument("--stretch_max", type=float, default=98,
                        help="对比度拉伸上限百分位 (0-100)，例如 98 表示忽略最高 2% 的值")
    parser.add_argument("--use_minmax", action="store_true", help="强制使用 Min-Max 归一化 (0-1) 而非百分位拉伸")

    args = parser.parse_args()

    # 1. 加载数据
    print(f"Loading data from {args.infer_csv}...")
    df = pd.read_csv(args.infer_csv)

    # 2. 数据透视：从列表转为矩阵 (Rasterization)
    # 假设数据是规则格网，利用 lat, lon 转为 2D 数组
    print("Rasterizing data...")
    # pivot_table 会自动处理排序，并将缺失值填为 NaN
    grid_matrix = df.pivot_table(index='lat', columns='lon', values='prob_mean')

    # 获取坐标范围
    lons = grid_matrix.columns.values
    lats = grid_matrix.index.values
    # extent = [xmin, xmax, ymin, ymax]
    # 注意：imshow 的 extent 需要半个像素的偏移才能对齐中心，这里简化处理直接用范围
    res_x = (lons.max() - lons.min()) / (len(lons) - 1)
    res_y = (lats.max() - lats.min()) / (len(lats) - 1)
    extent = [lons.min() - res_x / 2, lons.max() + res_x / 2,
              lats.min() - res_y / 2, lats.max() + res_y / 2]

    # 3. 计算对比度拉伸范围
    raw_values = df['prob_mean'].values
    if args.use_minmax:
        vmin, vmax = raw_values.min(), raw_values.max()
        print(f"Contrast Strategy: Min-Max [{vmin:.4f}, {vmax:.4f}]")
    else:
        # 百分位拉伸 (推荐)：忽略极端值，拉伸中间部分
        vmin = np.nanpercentile(raw_values, args.stretch_min)
        vmax = np.nanpercentile(raw_values, args.stretch_max)
        print(f"Contrast Strategy: Percentile ({args.stretch_min}-{args.stretch_max}%) [{vmin:.4f}, {vmax:.4f}]")

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # 绘制栅格 (imshow)
    # origin='lower' 因为纬度是从下到上增加的
    im = ax.imshow(grid_matrix,
                   extent=extent,
                   origin='lower',
                   cmap='RdYlBu_r',
                   vmin=vmin, vmax=vmax,  # 这里的 vmin/vmax 控制了颜色的映射范围
                   interpolation='nearest',  # 'nearest' 保持像素感，'bilinear' 会平滑
                   zorder=1)

    # 5. 叠加边界
    if args.boundary_file and os.path.exists(args.boundary_file):
        try:
            gdf_bound = gpd.read_file(args.boundary_file)
            if gdf_bound.crs and gdf_bound.crs.to_string() != "EPSG:4326":
                gdf_bound = gdf_bound.to_crs("EPSG:4326")
            gdf_bound.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=3)
        except Exception as e:
            print(f"Warning: Could not load boundary: {e}")

    # 6. 叠加现有站点 (可选：稍微调小一点，避免遮挡热力图)
    df_p = df[df['is_P'] == 1]
    ax.scatter(df_p['lon'], df_p['lat'],
               c='black', s=5, marker='+', linewidth=0.5, alpha=0.6,
               label='Existing Stations', zorder=2)

    # 7. 美化
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(f'Suitability Probability (Stretched [{vmin:.2f}, {vmax:.2f}])', fontsize=10)

    ax.set_title(args.title, fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper right', framealpha=0.8)

    # 设置显示范围与数据范围一致 (去除白边)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches='tight')
    print(f"Map saved to {args.out_png}")


if __name__ == "__main__":
    main()