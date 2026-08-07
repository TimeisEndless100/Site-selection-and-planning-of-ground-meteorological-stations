"""实验配置加载器：base.yaml（全局默认）+ 家族 YAML → 带默认值合并的配置字典。

解析器按官方 YAML 规范，不走第三方依赖（环境最小化）。结构约定（见各 configs/*.yaml）：

    base:            # 家族级默认
      <key>: <value>
    <exp_name>:      # 每个实验一个顶级键
      stage: cnn|gnn|decision
      cells: ...     # 该实验操纵的格子
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.registry import get

_CONFIG_DIR = Path(__file__).parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """浅递归合并：override 覆盖 base（标量直替，列表整体替换）。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_family(family: str) -> dict[str, Any]:
    """加载家族配置：合并 base.yaml + <family>.yaml。返回 {exp_name: merged_cfg}。"""
    with open(_CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(_CONFIG_DIR / f"{family}.yaml", encoding="utf-8") as f:
        family_cfg = yaml.safe_load(f)

    family_cfg.pop("family", None)  # 元信息，非实验
    result: dict[str, Any] = {}
    family_base = family_cfg.pop("base", {})
    merged_base = _deep_merge(base, family_base)
    for exp_name, cfg in family_cfg.items():
        if exp_name.startswith("_"):  # 下划线前缀 = 元信息/注释占位，跳过
            continue
        # 每个实验都是注册过的实验（防手滑新增未注册名字）
        meta = get(exp_name)
        merged = _deep_merge(merged_base, cfg)
        merged["exp"] = exp_name
        merged["category"] = meta.category
        merged["bucket"] = meta.bucket
        merged["section"] = meta.section
        merged["legacy_id"] = meta.legacy_id
        result[exp_name] = merged
    return result


def load_experiment(name: str) -> dict[str, Any]:
    """加载单个实验（含默认值合并）。"""
    family = {
        "cmp": "main",
        "abla": "ablations",
        "sens": "sensitivity",
        "proto": "protocol",
        "app": "application",
    }[name.split("_", 1)[0]]
    return load_family(family)[name]
