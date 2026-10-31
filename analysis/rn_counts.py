# -*- coding: utf-8 -*-
"""RN 阈值扫描：标签构成表（纯计算，零 GPU）。

背景：sens_rn_threshold 的物理机制是——阈值 (slope_thresh, water_thresh) 改变的是
**标签构成**（RN/U/P 三类计数），不是模型本身。因此"计数随阈值变化"这一交付物
可直接由 patches_info.csv 纯计算生成（秒级、零 GPU），无需为每个阈值训练模型。
训练配置只保留少量代表性阈值（见 sensitivity.yaml sens_rn_threshold.cells），
用于检验结论排序（如 PURN > PN）是否随阈值保持。

RN 判定规则与训练期严格一致（src/core/rn.assign_sets，physics 策略，严格 >）：

    is_rn = (slope_max > slope_thresh) | (water_ratio > water_thresh)
    set = P if is_P else (RN if is_rn else U)

用法：
    python -m analysis.rn_counts \
        --patches_csv data/processed/patches_info.csv \
        --out_dir <dir>

输出 rn_counts.csv（长表）：slope_thresh, water_thresh, n_P, n_RN, n_U, n_total,
non_rn_pos_rate（候选地内的正类率 = n_P / (n_total - n_RN)）。
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.core.rn import assign_sets

# 默认细网格（比训练配置更密，全图成本 ~秒级）。值可经 CLI 覆盖。
DEFAULT_SLOPE_GRID = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]
DEFAULT_WATER_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def count_table(df: pd.DataFrame, slope_grid: list[float], water_grid: list[float]) -> pd.DataFrame:
    """对每个 (slope_thresh, water_thresh) 计算 RN/U/P 计数，返回长表。"""
    rows: list[dict] = []
    n_total = len(df)
    for s, w in product(slope_grid, water_grid):
        out = assign_sets(
            df,
            rn_slope_thresh=float(s),
            rn_water_thresh=float(w),
            rn_strategy="physics",
            seed=42,
        )
        n_rn = int((out["set"] == "RN").sum())
        n_p = int((out["set"] == "P").sum())
        n_u = int((out["set"] == "U").sum())
        constructible = max(n_total - n_rn, 1)
        rows.append(
            {
                "slope_thresh": float(s),
                "water_thresh": float(w),
                "n_P": n_p,
                "n_RN": n_rn,
                "n_U": n_u,
                "n_total": n_total,
                "non_rn_pos_rate": round(n_p / constructible, 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RN 阈值扫描：标签构成表（纯计算，零 GPU）")
    parser.add_argument("--patches_csv", type=str, default="data/processed/patches_info.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/rn_counts")
    parser.add_argument("--slope_grid", type=str, default=",".join(f"{x:g}" for x in DEFAULT_SLOPE_GRID),
                        help="逗号分隔的 slope 阈值列表")
    parser.add_argument("--water_grid", type=str, default=",".join(f"{x:g}" for x in DEFAULT_WATER_GRID),
                        help="逗号分隔的 water 阈值列表")
    args = parser.parse_args()

    df = pd.read_csv(args.patches_csv)
    required = ["is_P", "slope_max", "water_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"patches_csv 缺列: {missing}")

    slope_grid = [float(x) for x in args.slope_grid.split(",") if x.strip()]
    water_grid = [float(x) for x in args.water_grid.split(",") if x.strip()]
    if not slope_grid or not water_grid:
        raise SystemExit("slope_grid / water_grid 不能为空")

    table = count_table(df, slope_grid, water_grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rn_counts.csv"
    table.to_csv(out_path, index=False)
    print(f"RN 阈值计数表已写入 {out_path}（{len(table)} 行）")
    # 打印默认配置 + 首末配置便于快速检查
    default = table[(table["slope_thresh"] == 45.0) & (table["water_thresh"] == 0.5)]
    if len(default) > 0:
        print("默认配置 (45.0, 0.5):")
        print(default.to_string(index=False))


if __name__ == "__main__":
    main()
