"""实验分区：语义化实验的声明层（注册表 + 配置）与家族模块。

2026-08-07 起废弃 E1–E19 模糊编号，统一语义化实验名（`cmp_/abla_/sens_/proto_/app_/diag_`）。
- 注册表：`experiments.registry`（语义名 → 分类/指标桶/论文章节/旧编号/状态）
- 配置：`experiments.configs/*.yaml`（每族一份，机器可读的实验格子）
- 家族模块：`experiments.<family>.py`（薄壳，声明实验格子；执行归 runner 层）

命名权威：`docs/实验地图与论文对照.md` §2。
"""
from experiments.registry import (
    Experiment,
    all_experiments,
    by_bucket,
    by_category,
    by_legacy,
    get,
)

__all__ = [
    "Experiment",
    "all_experiments",
    "by_bucket",
    "by_category",
    "by_legacy",
    "get",
]
