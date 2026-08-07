"""应用展示（论文 §4.6）：真实决策任务演示。

写作口径：应用展示的指标（rmclp_gain / 覆盖率）与预测指标（AUC 等）属于不同维度，
永不混比。app_infer 是前置步骤（生成候选源概率图），不单独成表。
⚠️ app_rmclp_gain / app_rmclp_vs_mclp 的上游决策链未闭合（约束/需求参数未定），
修复上游（P5 last-epoch、候选源质量）前结论不可写死。
"""
from __future__ import annotations

from typing import Any

from experiments.config import load_family


def plan() -> dict[str, dict[str, Any]]:
    """返回该族全部实验的合并配置：{app_infer, app_rmclp_gain, app_rmclp_vs_mclp}。"""
    return load_family("application")


def decision_chain() -> list[str]:
    """决策链：app_infer → app_rmclp_gain → app_rmclp_vs_mclp。"""
    return ["app_infer", "app_rmclp_gain", "app_rmclp_vs_mclp"]


if __name__ == "__main__":
    from pprint import pprint

    for name, cfg in plan().items():
        print(f"\n=== {name} (legacy {cfg['legacy_id']}, bucket={cfg['bucket']}) ===")
        pprint(cfg.get("cells"))
