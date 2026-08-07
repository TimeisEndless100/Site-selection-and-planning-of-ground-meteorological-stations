"""作业展开：experiments/configs/*.yaml → 作业清单（jobs.json）。

每个作业 = (exp, cell_label, seed, fold, config)，config 是合并默认值后的完整
参数字典（写入 outputs/<exp>/<config>/config.json，保证每个产物自带复现信息）。

Phase 3 目标：旧 run_closure_plan.py 的 add_*_job 全部由此取代；本文件现为骨架，
to_cli() 渲染的是 Phase 2 后的目标命令（runner.train.*），旧脚本仍走 scripts/main.py。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.config import load_family


def _expand_cells(cells: Any) -> list[dict]:
    """统一格子形状：list[dict] 原样返回；dict{参数: [值,...]} 展开为笛卡尔积。

    消融/对比配置用 list[dict]（每格一个开关组合），敏感性配置用 dict 网格。
    """
    if cells is None:
        return []
    if isinstance(cells, list):
        return [dict(c) for c in cells]
    if isinstance(cells, dict):
        keys = list(cells)
        if not keys:
            return []
        return [dict(zip(keys, value)) for value in itertools.product(*(cells[k] for k in keys))]
    raise TypeError(f"cells 结构不支持: {type(cells)}")


def build_jobs(family: str, seeds: Iterable[int] | None = None, folds: int | None = None) -> list[dict]:
    """展开一族实验为作业清单。"""
    cfgs = load_family(family)
    jobs: list[dict] = []
    for exp_name, cfg in cfgs.items():
        base_seeds = seeds or cfg.get("seeds", [42, 52, 62])
        base_folds = folds or cfg.get("folds", 5)
        for ci, cell in enumerate(_expand_cells(cfg.get("cells"))):
            label = cell.get("label") or f"{exp_name}_c{ci}"
            for seed in base_seeds:
                for fold in range(1, base_folds + 1):
                    job_cfg = {k: v for k, v in cfg.items() if k not in ("cells", "base")}
                    job_cfg.update(cell)
                    job_cfg.update({"seed": seed, "fold": fold})
                    jobs.append({
                        "exp": exp_name,
                        "cell": label,
                        "seed": seed,
                        "fold": fold,
                        "config": job_cfg,
                    })
    return jobs


def to_cli(job: dict) -> list[str]:
    """渲染单作业的命令行（Phase 2 后目标形态：runner.train.*）。

    现为骨架：stage=cnn/gnn/decision 映射到对应入口，实际参数装配在 Phase 2
    平移脚本时逐一接线（config.json 已含全部参数，无需重复解析）。
    """
    stage = job["config"].get("stage", "cnn")
    module = {"cnn": "runner.train.train_cnn",
              "gnn": "runner.train.train_gnn",
              "decision": "runner.train.infer"}[stage]
    return ["python", "-m", module,
            "--config", f"outputs/{job['exp']}/{job['cell']}/config.json"]


def write_plan(family: str, out: str | Path) -> Path:
    """展开并写入 jobs.json。返回路径。"""
    jobs = build_jobs(family)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"family": family, "jobs": jobs}, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="展开一族实验为作业清单")
    parser.add_argument("family", choices=["main", "ablations", "sensitivity", "protocol", "application"])
    parser.add_argument("--out", default=None)
    ns = parser.parse_args()

    out = write_plan(ns.family, ns.out or f"outputs/plans/plan_{ns.family}.json")
    jobs = build_jobs(ns.family)
    print(f"family={ns.family}: {len(jobs)} jobs → {out}")
    sys.exit(0)
