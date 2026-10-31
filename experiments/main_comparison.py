"""对比方法（论文主表 §4.2）：主方法 vs 基线方法。

薄壳模块：从配置 + 注册表声明实验配置，执行归 runner 层。
阶段①（CNN）与阶段②（GNN）分别对应一张主表；基线方法经统一管线复现，
共享同一数据划分 / 指标 / 显著性框架（P4 修复目标）。
"""
from __future__ import annotations

from typing import Any

from experiments.config import load_family


def plan() -> dict[str, dict[str, Any]]:
    """返回该族全部实验的合并配置：{cmp_main_cnn, cmp_main_gnn, cmp_pu_baseline, cmp_graph_baseline}。"""
    return load_family("main")


def cnn_cells(exp: str = "cmp_main_cnn") -> list[dict]:
    """阶段① CNN 对比配置（主方法 vs PN / TaylorPU）。"""
    return plan()[exp]["cells"]


def gnn_cells(exp: str = "cmp_main_gnn") -> list[dict]:
    """阶段② GNN 对比配置（主方法 vs GCN / GAT）。"""
    return plan()[exp]["cells"]


if __name__ == "__main__":
    from pprint import pprint

    for name, cfg in plan().items():
        print(f"\n=== {name} (legacy {cfg['legacy_id']}, bucket={cfg['bucket']}) ===")
        pprint(cfg["cells"])
