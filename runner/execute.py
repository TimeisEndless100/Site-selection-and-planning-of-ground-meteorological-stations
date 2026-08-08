# -*- coding: utf-8 -*-
"""作业执行：断点续跑 + 子进程日志 + 状态回写（← run_closure_plan.py main 的循环）。

由 runner/plan.py 渲染的矩阵驱动。断点续跑双保险：
- 矩阵行 status==success 且 result_file 存在；
- 或 <out_dir>/DONE 标记文件存在（作业完成时落标记）。

用法（通常不直接调用，由 plan.py --execute 委托）：
    python -m runner.execute --out_root outputs/...  --resume
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.matrix import merge_matrices

ROOT_DIR = Path(__file__).resolve().parents[1]


def done_marker(out_dir: str | Path) -> Path:
    return Path(out_dir) / "DONE"


def is_done(row: pd.Series) -> tuple[bool, str]:
    """断点判断：返回 (是否完成, 原因)。"""
    result_file = Path(str(row.get("result_file", "") or ""))
    if str(row.get("status", "")) == "success" and result_file.exists():
        return True, "matrix success + result_file 存在"
    if done_marker(str(row.get("out_dir", ""))).exists():
        return True, "DONE 标记存在"
    return False, ""


def run_jobs(
    df: pd.DataFrame,
    out_root: Path,
    run_label: str,
    stop_on_error: bool = False,
    cwd: Path | None = None,
) -> pd.DataFrame:
    """顺序执行矩阵作业，逐个回写矩阵（合并经 src/core/matrix.py）。"""
    cwd = cwd or ROOT_DIR
    matrix_csv = out_root / "experiment_matrix.csv"
    n_done = n_fail = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        is_done_flag, reason = is_done(row)
        if is_done_flag:
            print(f"[{idx + 1}/{len(df)}] {row['job_id']} -> skip ({reason})")
            n_done += 1
            continue

        command = str(row["command"])
        log_file = Path(str(row["log_file"]))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        with log_file.open("w", encoding="utf-8", errors="ignore") as f:
            f.write(f"COMMAND: {command}\n\n")
            proc = subprocess.run(command, shell=False, stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))
        runtime_min = (time.time() - start) / 60.0

        df.loc[idx, "runtime_min"] = round(runtime_min, 2)
        df.loc[idx, "return_code"] = proc.returncode
        if proc.returncode == 0:
            df.loc[idx, "status"] = "success"
            done_marker(str(row["out_dir"])).touch()
            n_done += 1
        else:
            df.loc[idx, "status"] = "failed"
            n_fail += 1
            if stop_on_error:
                _save(df, out_root, run_label, matrix_csv)
                print(f"Job failed: {row['job_id']} -> {row['exp_id']}. Stopped due to --stop_on_error.")
                return df

        _save(df, out_root, run_label, matrix_csv)
        print(f"[{idx + 1}/{len(df)}] {row['job_id']} -> {df.loc[idx, 'status']} ({runtime_min:.1f} min)")

    print(f"[execute] done={n_done} fail={n_fail} total={len(df)}")
    return df


def _save(df: pd.DataFrame, out_root: Path, run_label: str, matrix_csv: Path) -> None:
    """回写矩阵；不覆盖旧快照里更早的已完成状态（断点续跑合并）。"""
    existing = pd.read_csv(matrix_csv) if matrix_csv.exists() else None
    root_df = merge_matrices(existing, df) if existing is not None else df
    root_df.to_csv(matrix_csv, index=False)
    snapshot = out_root / f"experiment_matrix_{run_label}.csv"
    root_df.to_csv(snapshot, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="执行 plan 矩阵中的作业（断点续跑）")
    parser.add_argument("--out_root", default="outputs/exp_closure_20260315")
    parser.add_argument("--resume", action="store_true", help="跳过已完成作业（默认开启）")
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    matrix_csv = out_root / "experiment_matrix.csv"
    if not matrix_csv.exists():
        raise SystemExit(f"Matrix not found: {matrix_csv}（先运行 python -m runner.plan）")
    df = pd.read_csv(matrix_csv)
    if not args.resume:
        df = df.copy()
        df["status"] = "planned"  # 显式重跑全部
    run_label = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_jobs(df, out_root, run_label, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
