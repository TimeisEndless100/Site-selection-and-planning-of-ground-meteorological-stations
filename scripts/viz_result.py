# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_csv", required=True)
    parser.add_argument("--out_png", default="result_map.png")
    args = parser.parse_args()

    df = pd.read_csv(args.infer_csv)

    plt.figure(figsize=(10, 8))
    # 1. 画背景 (所有 U 点，用概率着色)
    plt.scatter(df['lon'], df['lat'], c=df['prob_mean'], cmap='RdYlBu_r', s=1, alpha=0.5, label='Suitability')
    plt.colorbar(label='Buildability Probability')

    # 2. 画现有站 (黑色点)
    existing = df[df['is_P'] == 1]
    plt.scatter(existing['lon'], existing['lat'], c='black', s=5, marker='x', alpha=0.5, label='Existing')

    plt.title("Meteorological Station Buildability Map (AI Predicted)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    plt.savefig(args.out_png, dpi=300)
    print(f"Map saved to {args.out_png}")


if __name__ == "__main__":
    main()