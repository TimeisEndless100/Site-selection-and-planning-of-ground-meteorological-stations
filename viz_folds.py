# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入你现有的切分函数
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from src.spatial_split import make_spatial_folds


def main():
    # 1. 加载数据
    info_csv = "data/processed/patches_info.csv"
    print(f"Loading {info_csv}...")
    df = pd.read_csv(info_csv)

    # 2. 运行与训练时完全一致的划分逻辑
    folds = make_spatial_folds(df, n_splits=5, grid_size=10, seed=42)

    # 3. 将 fold_id 写入 DataFrame 以供绘图
    df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        df.loc[val_idx, 'fold'] = fold_idx

    # 4. 绘制 Fold 分布图
    plt.figure(figsize=(10, 8), dpi=300)
    cmap = plt.get_cmap('Set1', 5)  # 使用离散的 5 种颜色

    sc = plt.scatter(df['lon'], df['lat'], c=df['fold'], cmap=cmap, s=2, alpha=0.8, marker='s')

    # 添加颜色条，标明 Fold 0-4
    cbar = plt.colorbar(sc, ticks=range(5), fraction=0.03, pad=0.04)
    cbar.set_label('Fold ID (Validation Set)')

    plt.title('Spatial Block Cross-Validation (10x10 Grid)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 保持地图比例
    plt.gca().set_aspect('equal')

    out_png = "outputs/spatial_folds_map.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Fold map saved to {out_png}")


if __name__ == "__main__":
    main()