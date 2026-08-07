# -*- coding: utf-8 -*-
"""
Unified project entrypoint.
This keeps legacy scripts working while offering a single explicit CLI surface.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "train-cnn": "run_train.py",
    "train-gnn": "run_train_gnn.py",
    "train-ssl": "run_train_ssl.py",
    "train-supcon": "run_train_ssl.py",
    "train-contrastive": "run_train_ssl.py",
    "infer-cnn": "run_infer.py",
    "infer-gnn-all": "run_infer_gnn_all.py",
    "rmclp": "run_rmclp.py",
    "external-baselines": "run_external_baselines.py",
    "rmclp-curve": "run_rmclp_curve.py",
    "fold4-analysis": "run_fold4_analysis.py",
    "closure-plan": "run_closure_plan.py",
    "closure-aggregate": "run_closure_aggregate.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(SCRIPT_MAP.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to target script")
    ns = parser.parse_args()

    command = ns.command
    extra_args = []
    # Legacy aliases kept at command level without extra duplicate scripts.
    if command == "train-supcon":
        command = "train-ssl"
        extra_args = ["--ssl_mode", "supcon"]
    elif command == "train-contrastive":
        command = "train-ssl"
        extra_args = ["--ssl_mode", "simclr"]

    script = Path(__file__).parent / SCRIPT_MAP[command]
    cmd = [sys.executable, str(script)] + extra_args + ns.args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
