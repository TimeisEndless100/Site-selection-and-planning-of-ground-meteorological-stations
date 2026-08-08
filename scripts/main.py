# -*- coding: utf-8 -*-
"""
Unified project entrypoint (临时桥接壳).

Phase E 将删除本文件：所有命令改用 `python -m <module>` 直连。
在删除前，它把旧命令名转发到 runner/analysis 的新模块入口，保证
run_closure_plan.py 等尚未迁移的脚本继续可用。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 旧命令名 → (模块, 前缀参数)。模块经 `python -m` 调用；
# closure-plan / closure-aggregate 是待拆解的脚本，仍走绝对脚本路径。
SCRIPT_MAP = {
    "train-cnn": ("runner.train.train_cnn", []),
    "train-gnn": ("runner.train.train_gnn", []),
    "train-ssl": ("runner.train.train_ssl", []),
    "train-supcon": ("runner.train.train_ssl", ["--ssl_mode", "supcon"]),
    "train-contrastive": ("runner.train.train_ssl", ["--ssl_mode", "simclr"]),
    "infer-cnn": ("runner.train.infer", []),
    "infer-gnn-all": ("runner.train.infer_gnn_all", []),
    "rmclp": ("runner.train.rmclp", ["single"]),
    "external-baselines": ("runner.train.baselines", []),
    "rmclp-curve": ("runner.train.rmclp", ["curve"]),
    "fold4-analysis": ("analysis.fold4", []),
    "closure-plan": (str(Path(__file__).parent / "run_closure_plan.py"), []),
    "closure-aggregate": ("analysis.aggregate", []),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(SCRIPT_MAP.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to target module")
    ns = parser.parse_args()

    target, prefix = SCRIPT_MAP[ns.command]
    if target.endswith(".py"):
        cmd = [sys.executable, target] + prefix + ns.args
    else:
        cmd = [sys.executable, "-m", target] + prefix + ns.args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
