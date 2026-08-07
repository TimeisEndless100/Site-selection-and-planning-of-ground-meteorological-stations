"""消融实验（论文 §4.3）：删/换一个明确定义的组件。

预算纪律：消融内所有格子训练预算统一（budget_epochs=20），禁止 epoch 数混杂
（历史缺陷 P1：No-warmup 格只有 15 epoch）。每个格子与锚点（cmp_main_cnn/gnn
的主方法点）只差一个组件开关。
"""
from __future__ import annotations

from typing import Any

from experiments.config import load_family


def plan() -> dict[str, dict[str, Any]]:
    """返回该族全部实验的合并配置：{abla_rn_source, abla_objective, abla_graph, abla_structure}。"""
    return load_family("ablations")


def rn_source_cells() -> list[dict]:
    """RN 来源策略格子（physics / random / none / noisy10/20/30）。"""
    return plan()["abla_rn_source"]["cells"]


def objective_cells() -> list[dict]:
    """目标组件格子（warmup × nnPU × RN penalty，含待补的 Supervised 格）。"""
    return plan()["abla_objective"]["cells"]


def graph_cells() -> list[dict]:
    """图传播格子（NodeMLP / SAGE 全图 / SAGE 自环图）。"""
    return plan()["abla_graph"]["cells"]


if __name__ == "__main__":
    from pprint import pprint

    for name, cfg in plan().items():
        print(f"\n=== {name} (legacy {cfg['legacy_id']}, bucket={cfg['bucket']}) ===")
        pprint(cfg.get("cells"))
        if "note" in cfg:
            print(f"note: {cfg['note']}")
