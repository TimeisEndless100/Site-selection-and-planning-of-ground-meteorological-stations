"""敏感性实验（论文 §4.4）：连续参数扫描，测结论稳健性。

写作口径：敏感性 = 稳健性展示（非"选最优"）。扫描结论读的是
「结论排序是否保持」「指标变化区间」，不是某配置指标比参照点高。
阈值类实验（sens_rn_threshold）每档必须附 RN/U/P 计数，暴露数据构成混杂。
"""
from __future__ import annotations

from typing import Any

from experiments.config import load_family


def plan() -> dict[str, dict[str, Any]]:
    """返回该族全部实验的合并配置：{sens_rn_threshold, sens_pi, sens_lambda, sens_graph_k, sens_fold_grid, sens_feature}。"""
    return load_family("sensitivity")


def cell_values(exp: str) -> dict[str, list]:
    """返回某敏感性实验的扫描配置（参数名 → 扫描值列表）。"""
    return plan()[exp]["cells"]


if __name__ == "__main__":
    from pprint import pprint

    for name, cfg in plan().items():
        print(f"\n=== {name} (legacy {cfg['legacy_id']}, bucket={cfg['bucket']}) ===")
        pprint(cfg["cells"])
