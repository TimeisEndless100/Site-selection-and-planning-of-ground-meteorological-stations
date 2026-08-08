"""结果聚合层（论文「结果」章）：把 outputs/ 产物整理成统计口径统一的表格。

- aggregate.py          聚合入口（← scripts/run_closure_aggregate.py）：读 experiment_matrix.csv，
                        按 result_type 装观测 → master_results.csv / master_summary.csv /
                        stats_results.csv / negative_results.csv / reproducibility_checklist.csv /
                        plot_ready_index.csv
- statistics.py         Wilcoxon / Holm / Cohen's d / bootstrap 单一实现（所有报告复用）
- data_composition.py   RN/U/P 计数报告（阈值类敏感性实验的数据构成混杂显式化）

设计原则（docs/代码重构与分区计划.md §0.4）：统计口径单一实现，禁止各脚本各自实现一份；
矩阵合并走 src/core/matrix.py（与 runner/plan.py 共享）。
"""
