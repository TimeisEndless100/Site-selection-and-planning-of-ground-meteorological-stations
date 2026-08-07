"""论文表格：只消费 analysis/aggregate.py 的 results_by_config.csv。

表结构（docs/实验地图与论文对照.md §5）：
- 主结果表（桶①、桶②各一张）：模型 | AUC | AUPRC | Brier | Recall@K | Δ vs 主方法 | p_adj
- 消融表：组件开关（✓/✗，每行只差一格）| 指标×4 | Δ vs Full | p_adj
- 敏感性表：参数 | 扫描值 | 指标变化区间 | RN/U/P 计数（阈值类）| 结论排序是否保持
- 协议表：协议 | AUC（mean±std）| SAGE > GAT > GCN 排序是否保持

本文件为骨架。
"""
from __future__ import annotations


def main_table(bucket: str) -> None:
    """生成主结果表（bucket=cnn 或 gnn）。"""
    raise NotImplementedError("Phase 3 实现：消费 results_by_config.csv")


def ablation_table() -> None:
    raise NotImplementedError("Phase 3 实现")


def sensitivity_table() -> None:
    raise NotImplementedError("Phase 3 实现")


def protocol_table() -> None:
    raise NotImplementedError("Phase 3 实现")


if __name__ == "__main__":
    main_table("cnn")
