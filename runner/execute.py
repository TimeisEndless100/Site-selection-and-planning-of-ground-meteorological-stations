"""作业执行：串/并行 + 断点续跑 + 重试（骨架）。

Phase 3 后由 runner/plan.py 的 jobs.json 驱动。断点续跑 = 每个作业完成时在
outputs/<exp>/<config>/DONE 落标记，重跑时跳过已有标记的作业。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from runner.plan import to_cli


def load_jobs(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["jobs"]


def _done_marker(job: dict) -> Path:
    return Path("outputs") / job["exp"] / job["cell"] / "DONE"


def run_plan(jobs: list[dict], run_cmd: Callable[[list[str]], int],
             resume: bool = True, start_at: int = 0) -> None:
    """顺序执行作业。run_cmd 注入真实执行器（便于测试与换并行后端）。"""
    n_done = n_fail = 0
    for i, job in enumerate(jobs):
        if i < start_at:
            continue
        if resume and _done_marker(job).exists():
            continue
        rc = run_cmd(to_cli(job))
        if rc == 0:
            _done_marker(job).touch()
            n_done += 1
        else:
            n_fail += 1
            print(f"[execute] FAIL job {i}: exp={job['exp']} cell={job['cell']} seed={job['seed']} fold={job['fold']}")
    print(f"[execute] done={n_done} fail={n_fail} total={len(jobs)}")
