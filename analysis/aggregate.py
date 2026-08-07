"""按 (exp_id, config, method) 分组聚合（Phase 1 目标，修复 P3）。

现状缺陷：run_closure_aggregate.py 按 (stage, model) 输出 master_summary.csv，
E1–E4 组内对照（physics vs random vs none vs noisy）被压成一行"PURN-CNN"。
改后：每行 = (exp_id, config, method)，n=15 观测（3 seed × 5 fold），输出
results_by_config.csv；统计检验走 analysis/statistics.py。

本文件为骨架。实现时：
- exp_id 元信息（分类/桶/章节）由 experiments/registry.py 提供 → 自动打 bucket 标签，
  保证三桶（①CNN ②GNN ③下游决策）永不混表。
- E5 + E6 同表（P4）：E6 与 E5 在同一分组域内做配对 Wilcoxon + Holm。
"""
from __future__ import annotations


def aggregate() -> None:
    """重写点：按 (exp_id, config, method) 聚合 outputs/ 产物 → results_by_config.csv。"""
    raise NotImplementedError("Phase 1 实现：对齐旧 master_summary.csv 做差分测试后启用")


if __name__ == "__main__":
    aggregate()
