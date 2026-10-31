"""数据构成报告：RN/U/P 三类样本计数随阈值/先验变化（E2/E3 混杂显式化）。

背景：sens_rn_threshold（RN 阈值 slope×water）与 sens_pi（类先验 π）改变的是
**标签构成**（哪些点被判为 RN / U / P），不只是模型参数。若只看指标变化区间，
会误读为"模型敏感"，实际是"数据构成变了"。每档必须附三计数，聚合时进正式输出。

本文件为骨架。
"""
from __future__ import annotations


def report_counts(cfg_dir: str) -> dict:
    """读一档实验的配置 + 数据，返回 {slope, water, n_rn, n_u, n_p}。"""
    raise NotImplementedError("Phase 4 实现：与 sens_rn_threshold 运行挂钩")


if __name__ == "__main__":
    report_counts()
