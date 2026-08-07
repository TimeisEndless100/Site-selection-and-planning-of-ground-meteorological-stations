"""结果聚合层（论文「结果」章）：把 outputs/ 产物整理成统计口径统一的表格。

- aggregate.py          按 (exp_id, config, method) 分组聚合 → results_by_config.csv
                        （修复 P3：不再按 stage 压组丢失 E1–E4 组内对照）
- statistics.py         Wilcoxon / Holm / Cohen's d / bootstrap 单一实现（所有报告复用）
- data_composition.py   RN/U/P 计数报告（阈值类敏感性实验的数据构成混杂显式化）

设计原则（docs/代码重构与分区计划.md §0.4）：统计口径单一实现，禁止各脚本各自实现一份。
"""
