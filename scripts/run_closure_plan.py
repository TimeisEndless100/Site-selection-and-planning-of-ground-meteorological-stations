# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class Job:
    job_id: str
    exp_id: str
    stage: str
    model: str
    seed: int
    command: list[str]
    out_dir: str
    result_type: str
    result_file: str
    negative_file: str = ""


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_name_list(values: str | Iterable[str]) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    out: list[str] = []
    for value in raw_values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def should_keep(job: Job, tokens: set[str]) -> bool:
    if not tokens:
        return True
    keys = {
        job.exp_id,
        job.stage,
        job.model,
        job.exp_id.split("_")[0],
    }
    return any(token in keys or job.exp_id.startswith(token) for token in tokens)


def list2cmdline(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def sanitize_run_token(text: str) -> str:
    text = text.strip()
    if not text:
        return "all"
    chars: list[str] = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    token = "".join(chars)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "all"


def build_run_label(subset: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{sanitize_run_token(subset)}_{stamp}"


def ensure_job_key(df: pd.DataFrame) -> pd.DataFrame:
    if "job_key" in df.columns:
        return df.copy()
    out = df.copy()
    keys: list[str] = []
    for idx, row in out.iterrows():
        result_file = str(row.get("result_file", "") or "")
        command = str(row.get("command", "") or "")
        stage = str(row.get("stage", "") or "")
        exp_id = str(row.get("exp_id", "") or "")
        key = result_file or command or f"{stage}:{exp_id}:{idx}"
        keys.append(key)
    out["job_key"] = keys
    return out


def prefer_current_row(existing: dict, current: dict) -> dict:
    current_status = str(current.get("status", "planned") or "planned")
    existing_status = str(existing.get("status", "planned") or "planned")
    if current_status != "planned":
        return current
    if existing_status in {"", "planned", "nan", "None"}:
        return current
    return existing


def merge_plan_matrix(existing_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    existing_df = ensure_job_key(existing_df)
    current_df = ensure_job_key(current_df)
    all_cols = list(dict.fromkeys(existing_df.columns.tolist() + current_df.columns.tolist()))
    existing_df = existing_df.reindex(columns=all_cols)
    current_df = current_df.reindex(columns=all_cols)

    merged: list[dict] = []
    key_to_idx: dict[str, int] = {}

    for _, row in existing_df.iterrows():
        rec = row.to_dict()
        key = str(rec["job_key"])
        key_to_idx[key] = len(merged)
        merged.append(rec)

    for _, row in current_df.iterrows():
        rec = row.to_dict()
        key = str(rec["job_key"])
        if key in key_to_idx:
            merged[key_to_idx[key]] = prefer_current_row(merged[key_to_idx[key]], rec)
        else:
            key_to_idx[key] = len(merged)
            merged.append(rec)

    return pd.DataFrame(merged, columns=all_cols)


def save_plan_outputs(
    current_df: pd.DataFrame,
    matrix_csv: Path,
    matrix_csv_snapshot: Path,
    cmd_file: Path,
    cmd_file_snapshot: Path,
) -> pd.DataFrame:
    current_df = ensure_job_key(current_df)
    current_df.to_csv(matrix_csv_snapshot, index=False)

    if matrix_csv.exists():
        existing_df = pd.read_csv(matrix_csv)
        root_df = merge_plan_matrix(existing_df, current_df)
    else:
        root_df = current_df.copy()

    root_df.to_csv(matrix_csv, index=False)
    cmd_file.write_text("\n".join(root_df["command"].astype(str).tolist()) + "\n", encoding="utf-8")
    cmd_file_snapshot.write_text("\n".join(current_df["command"].astype(str).tolist()) + "\n", encoding="utf-8")
    return root_df


def add_cnn_job(
    jobs: list[Job],
    py: str,
    args: argparse.Namespace,
    exp_id: str,
    stage: str,
    model: str,
    seed: int,
    extra: list[str],
) -> None:
    out_dir = Path(args.out_root) / stage / exp_id / f"seed_{seed}"
    # 预算统一（E13 修复）：disable_warmup 时无 warm 阶段，nnpu_epochs 须补满
    # warm_epochs + nnpu_epochs 的总预算（默认 5+15=20），否则 no-warmup 格只训 15
    # epoch，与其他格出现 epoch 数混杂（历史缺陷 P1）。
    nnpu_epochs = args.cnn_nnpu_epochs
    if "--disable_warmup" in extra:
        nnpu_epochs += args.cnn_warm_epochs
    cmd = [
        py,
        "scripts/main.py",
        "train-cnn",
        "--processed_dir",
        args.processed_dir,
        "--out_dir",
        str(out_dir),
        "--mode",
        "pu",
        "--folds",
        str(args.folds),
        "--spatial_grid_size",
        str(args.grid_size),
        "--batch_size",
        str(args.cnn_batch_size),
        "--warm_epochs",
        str(args.cnn_warm_epochs),
        "--nnpu_epochs",
        str(nnpu_epochs),
        "--lr",
        str(args.cnn_lr),
        "--seed",
        str(seed),
    ]
    if args.fast_2fold:
        cmd.extend(["--fold_ids", "0,1"])
    if args.buffer_cells > 0:
        cmd.extend(["--buffer_cells", str(args.buffer_cells)])
    cmd.extend(extra)

    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id=exp_id,
            stage=stage,
            model=model,
            seed=seed,
            command=cmd,
            out_dir=str(out_dir),
            result_type="cnn_results",
            result_file=str(out_dir / "results.csv"),
        )
    )


def add_gnn_job(
    jobs: list[Job],
    py: str,
    args: argparse.Namespace,
    exp_id: str,
    stage: str,
    model: str,
    seed: int,
    gnn_type: str,
    eval_mode: str,
    out_base: Path | None = None,
    extra: list[str] | None = None,
) -> None:
    out_dir = (out_base if out_base is not None else Path(args.out_root) / stage / exp_id / f"seed_{seed}").resolve()
    cmd = [
        py,
        "scripts/main.py",
        "train-gnn",
        "--gnn_data_dir",
        args.gnn_data_dir,
        "--out_dir",
        str(out_dir),
        "--gnn_type",
        gnn_type,
        "--eval_mode",
        eval_mode,
        "--folds",
        str(args.folds),
        "--spatial_grid_size",
        str(args.grid_size),
        "--epochs",
        str(args.gnn_epochs),
        "--lr",
        str(args.gnn_lr),
        "--seed",
        str(seed),
        "--rn_slope_thresh",
        str(args.rn_slope_default),
        "--rn_water_thresh",
        str(args.rn_water_default),
        "--lambda_rn",
        str(args.lambda_default),
    ]
    if args.fast_2fold:
        cmd.extend(["--fold_ids", "0,1"])
    if args.buffer_cells > 0:
        cmd.extend(["--buffer_cells", str(args.buffer_cells)])
    if extra:
        cmd.extend(extra)

    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id=exp_id,
            stage=stage,
            model=model,
            seed=seed,
            command=cmd,
            out_dir=str(out_dir),
            result_type="gnn_results",
            result_file=str(out_dir / gnn_type / "results.csv"),
        )
    )


def add_rmclp_curve_job(
    jobs: list[Job],
    py: str,
    exp_id: str,
    stage: str,
    model: str,
    seed: int,
    out_dir: Path,
    processed_dir: str,
    demand_csv: str,
    p_add_grid: str,
    radius_grid: str,
    prob_grid: str,
    source_csv: list[str],
    selection_modes: str = "rmclp",
    rn_filter_modes: str = "true",
    lambda_disp: float = 0.3,
    skip_default_sources: bool = True,
) -> None:
    cmd = [
        py,
        "scripts/main.py",
        "rmclp-curve",
        "--processed_dir",
        processed_dir,
        "--demand_csv",
        demand_csv,
        "--out_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--p_add_grid",
        p_add_grid,
        "--radius_grid",
        radius_grid,
        "--prob_grid",
        prob_grid,
        "--selection_modes",
        selection_modes,
        "--rn_filter_modes",
        rn_filter_modes,
        "--lambda_disp",
        str(lambda_disp),
    ]
    if skip_default_sources:
        cmd.append("--skip_default_sources")
    for item in source_csv:
        cmd.extend(["--source_csv", item])

    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id=exp_id,
            stage=stage,
            model=model,
            seed=seed,
            command=cmd,
            out_dir=str(out_dir),
            result_type="rmclp_curve",
            result_file=str(out_dir / "rmclp_curve.csv"),
        )
    )


def build_jobs(args: argparse.Namespace) -> list[Job]:
    py = sys.executable
    seeds = parse_int_list(args.seeds)
    ext_seeds = parse_int_list(args.external_seeds)
    tokens = set(parse_name_list(args.subset))
    jobs: list[Job] = []

    # E1: RN removal/replacement/noise.
    e1_settings = [
        ("E1_rn_physics", "PURN-CNN", ["--rn_strategy", "physics"]),
        ("E1_rn_none", "PURN-CNN", ["--rn_strategy", "none"]),
        ("E1_rn_random", "PURN-CNN", ["--rn_strategy", "random"]),
        ("E1_rn_noisy10", "PURN-CNN", ["--rn_strategy", "noisy", "--rn_noise_rate", "0.1"]),
        ("E1_rn_noisy20", "PURN-CNN", ["--rn_strategy", "noisy", "--rn_noise_rate", "0.2"]),
        ("E1_rn_noisy30", "PURN-CNN", ["--rn_strategy", "noisy", "--rn_noise_rate", "0.3"]),
    ]
    for exp_prefix, model, extra in e1_settings:
        for seed in seeds:
            add_cnn_job(
                jobs,
                py,
                args,
                exp_id=f"{exp_prefix}_s{seed}",
                stage="E1",
                model=model,
                seed=seed,
                extra=extra + ["--rn_slope_thresh", str(args.rn_slope_default), "--rn_water_thresh", str(args.rn_water_default)],
            )

    # E2: RN threshold sensitivity.
    for slope in parse_float_list(args.e2_slope_grid):
        for water in parse_float_list(args.e2_water_grid):
            for seed in seeds:
                add_cnn_job(
                    jobs,
                    py,
                    args,
                    exp_id=f"E2_thr_s{slope:g}_w{water:g}_seed{seed}",
                    stage="E2",
                    model="PURN-CNN",
                    seed=seed,
                    extra=[
                        "--rn_strategy",
                        "physics",
                        "--rn_slope_thresh",
                        str(slope),
                        "--rn_water_thresh",
                        str(water),
                    ],
                )

    # E3: pi bias robustness.
    pi_ref = float(args.pi_ref)
    pi_grid = [
        ("estimated", None),
        ("pi_ref", pi_ref),
        ("pi_m20", pi_ref * 0.8),
        ("pi_p20", pi_ref * 1.2),
        ("pi_m50", pi_ref * 0.5),
        ("pi_p50", pi_ref * 1.5),
    ]
    for pi_tag, pi_val in pi_grid:
        for seed in seeds:
            extra = [
                "--rn_strategy",
                "physics",
                "--rn_slope_thresh",
                str(args.rn_slope_default),
                "--rn_water_thresh",
                str(args.rn_water_default),
            ]
            if pi_val is not None:
                extra.extend(["--pi_prior", str(max(0.01, min(0.5, pi_val)))])
            add_cnn_job(
                jobs,
                py,
                args,
                exp_id=f"E3_{pi_tag}_seed{seed}",
                stage="E3",
                model="PURN-CNN",
                seed=seed,
                extra=extra,
            )

    # E4: lambda_rn scan.
    for lam in parse_float_list(args.e4_lambda_grid):
        for seed in seeds:
            add_cnn_job(
                jobs,
                py,
                args,
                exp_id=f"E4_lambda_{lam:g}_seed{seed}",
                stage="E4",
                model="PURN-CNN",
                seed=seed,
                extra=[
                    "--rn_strategy",
                    "physics",
                    "--rn_slope_thresh",
                    str(args.rn_slope_default),
                    "--rn_water_thresh",
                    str(args.rn_water_default),
                    "--lambda_rn",
                    str(lam),
                ],
            )

    # E5: internal baselines (CNN + GNN).
    for mode, tag in [("pn", "PN-CNN"), ("taylor", "TaylorPU-CNN"), ("pu", "PURN-CNN")]:
        for seed in seeds:
            out_dir = Path(args.out_root) / "E5" / f"{tag}_seed{seed}"
            cmd = [
                py,
                "scripts/main.py",
                "train-cnn",
                "--processed_dir",
                args.processed_dir,
                "--out_dir",
                str(out_dir),
                "--mode",
                mode,
                "--folds",
                str(args.folds),
                "--spatial_grid_size",
                str(args.grid_size),
                "--batch_size",
                str(args.cnn_batch_size),
                "--warm_epochs",
                str(args.cnn_warm_epochs),
                "--nnpu_epochs",
                str(args.cnn_nnpu_epochs),
                "--lr",
                str(args.cnn_lr),
                "--seed",
                str(seed),
                "--rn_strategy",
                "physics",
                "--rn_slope_thresh",
                str(args.rn_slope_default),
                "--rn_water_thresh",
                str(args.rn_water_default),
                "--lambda_rn",
                str(args.lambda_default),
            ]
            if args.fast_2fold:
                cmd.extend(["--fold_ids", "0,1"])
            if args.buffer_cells > 0:
                cmd.extend(["--buffer_cells", str(args.buffer_cells)])

            jobs.append(
                Job(
                    job_id=f"job_{len(jobs):04d}",
                    exp_id=f"E5_{mode}_seed{seed}",
                    stage="E5",
                    model=tag,
                    seed=seed,
                    command=cmd,
                    out_dir=str(out_dir),
                    result_type="cnn_results",
                    result_file=str(out_dir / "results.csv"),
                )
            )

    # E5 GNN transductive for GCN/GAT/SAGE.
    for seed in seeds:
        base_dir = (Path(args.out_root) / "E5" / f"GNN_transductive_seed{seed}").resolve()
        for gnn_type, model_name in [("gcn", "GCN"), ("gat", "GAT"), ("sage", "SAGE")]:
            add_gnn_job(
                jobs,
                py,
                args,
                exp_id=f"E5_{gnn_type}_seed{seed}",
                stage="E5",
                model=model_name,
                seed=seed,
                gnn_type=gnn_type,
                eval_mode="transductive",
                out_base=base_dir,
            )

    # E13: PURN loss-component ablation.
    e13_settings = [
        ("full", "PURN-full", []),
        ("warm_nnpu", "PURN-warm+nnPU", ["--lambda_rn", "0"]),
        ("nnpu_rn", "PURN-nnPU+RN", ["--disable_warmup"]),
        ("nnpu_only", "PURN-nnPU-only", ["--disable_warmup", "--lambda_rn", "0"]),
    ]
    for tag, model_name, extra_flags in e13_settings:
        for seed in seeds:
            extra = [
                "--rn_strategy",
                "physics",
                "--rn_slope_thresh",
                str(args.rn_slope_default),
                "--rn_water_thresh",
                str(args.rn_water_default),
            ]
            if "--lambda_rn" not in extra_flags:
                extra.extend(["--lambda_rn", str(args.lambda_default)])
            add_cnn_job(
                jobs,
                py,
                args,
                exp_id=f"E13_{tag}_seed{seed}",
                stage="E13",
                model=model_name,
                seed=seed,
                extra=extra + extra_flags,
            )

    # E14: No-graph vs graph reasoning.
    e14_archs = [x.strip() for x in args.e14_archs.split(",") if x.strip()]
    for seed in seeds:
        base_dir = (Path(args.out_root) / "E14" / f"graph_ablation_seed{seed}").resolve()
        for arch in e14_archs:
            model_name = "NodeMLP" if arch == "mlp" else arch.upper()
            add_gnn_job(
                jobs,
                py,
                args,
                exp_id=f"E14_{arch}_seed{seed}",
                stage="E14",
                model=model_name,
                seed=seed,
                gnn_type=arch,
                eval_mode="transductive",
                out_base=base_dir,
            )
        if "sage" in e14_archs:
            # abla_graph 自环对照：SAGE 同架构无传播（--self_loop_only），与 NodeMLP 区分架构差异
            # 注意：out_base 必须独立于 graph_ablation_seed62——run_train_gnn 内部会拼 /{gnn_type}，
            # 若共用 base 会因 gnn_type 同为 sage 而覆盖主 SAGE 格。
            add_gnn_job(
                jobs,
                py,
                args,
                exp_id=f"E14_sage_selfloop_seed{seed}",
                stage="E14",
                model="SAGE-selfloop",
                seed=seed,
                gnn_type="sage",
                eval_mode="transductive",
                out_base=(Path(args.out_root) / "E14" / f"selfloop_seed{seed}").resolve(),
                extra=["--self_loop_only"],
            )

    # E6 external baselines.
    ext_out_dir = Path(args.out_root) / "E6_external"
    ext_cmd = [
        py,
        "scripts/main.py",
        "external-baselines",
        "--gnn_data_dir",
        args.gnn_data_dir,
        "--out_dir",
        str(ext_out_dir),
        "--folds",
        str(args.folds),
        "--grid_size",
        str(args.grid_size),
        "--seeds",
        ",".join(str(x) for x in ext_seeds),
        "--rn_slope_thresh",
        str(args.rn_slope_default),
        "--rn_water_thresh",
        str(args.rn_water_default),
        "--upu_epochs",
        str(args.upu_epochs),
        "--appnp_epochs",
        str(args.appnp_epochs),
        "--lp_max_iter",
        str(args.lp_max_iter),
        "--device",
        args.device,
    ]
    if args.fast_2fold:
        ext_cmd.extend(["--fold_ids", "0,1"])
    if args.buffer_cells > 0:
        ext_cmd.extend(["--buffer_cells", str(args.buffer_cells)])
    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E6_external",
            stage="E6",
            model="uPU+ElkanNoto+BaggingPU+LP+APPNP",
            seed=-1,
            command=ext_cmd,
            out_dir=str(ext_out_dir),
            result_type="external_results",
            result_file=str(ext_out_dir / "results_external.csv"),
            negative_file=str(ext_out_dir / "negative_external.csv"),
        )
    )

    # E7 transductive vs inductive (default SAGE).
    e7_archs = [x.strip() for x in args.e7_archs.split(",") if x.strip()]
    for seed in seeds:
        for arch in e7_archs:
            for eval_mode in ["transductive", "inductive"]:
                add_gnn_job(
                    jobs,
                    py,
                    args,
                    exp_id=f"E7_{arch}_{eval_mode}_seed{seed}",
                    stage="E7",
                    model=f"{arch.upper()}-{eval_mode}",
                    seed=seed,
                    gnn_type=arch,
                    eval_mode=eval_mode,
                    out_base=(Path(args.out_root) / "E7" / f"{arch}_{eval_mode}_seed{seed}").resolve(),
                )
            if args.e7_buffer_cells > 0:
                add_gnn_job(
                    jobs,
                    py,
                    args,
                    exp_id=f"E7_{arch}_buffered_seed{seed}",
                    stage="E7",
                    model=f"{arch.upper()}-buffered",
                    seed=seed,
                    gnn_type=arch,
                    eval_mode="inductive",
                    out_base=(Path(args.out_root) / "E7" / f"{arch}_buffered_seed{seed}").resolve(),
                    extra=["--buffer_cells", str(args.e7_buffer_cells)],
                )

    # Inference for closure chain (primary seed only).
    primary_seed = seeds[0]
    e5_pn_dir = Path(args.out_root) / "E5" / f"PN-CNN_seed{primary_seed}"
    e5_purn_dir = Path(args.out_root) / "E5" / f"PURN-CNN_seed{primary_seed}"
    e5_gnn_dir = Path(args.out_root) / "E5" / f"GNN_transductive_seed{primary_seed}"

    infer_jobs = [
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E9_infer_cnn",
            stage="E9",
            model="CNN-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-cnn",
                "--processed_dir",
                args.processed_dir,
                "--exp_dir",
                str(e5_pn_dir),
                "--out_name",
                "inference_results.csv",
            ],
            out_dir=str(e5_pn_dir),
            result_type="inference",
            result_file=str(e5_pn_dir / "inference_results.csv"),
        ),
        Job(
            job_id=f"job_{len(jobs) + 1:04d}",
            exp_id="E9_infer_purn",
            stage="E9",
            model="PURN-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-cnn",
                "--processed_dir",
                args.processed_dir,
                "--exp_dir",
                str(e5_purn_dir),
                "--out_name",
                "inference_results.csv",
            ],
            out_dir=str(e5_purn_dir),
            result_type="inference",
            result_file=str(e5_purn_dir / "inference_results.csv"),
        ),
        Job(
            job_id=f"job_{len(jobs) + 2:04d}",
            exp_id="E9_infer_gnn",
            stage="E9",
            model="PURN+GNN-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-gnn-all",
                "--gnn_data_dir",
                args.gnn_data_dir,
                "--demand_csv",
                args.demand_csv,
                "--base_exp_dir",
                str(e5_gnn_dir),
                "--archs",
                "sage",
                "gcn",
                "gat",
                "--p_add",
                "100",
                "--prob_thresh",
                "0.7",
            ],
            out_dir=str(e5_gnn_dir),
            result_type="inference",
            result_file=str(e5_gnn_dir / "inference_results_sage.csv"),
        ),
    ]
    jobs.extend(infer_jobs)

    # E7 inference for Fold4 uncertainty analysis (primary seed).
    e7_arch = e7_archs[0]
    e7_trans_seed_dir = Path(args.out_root) / "E7" / f"{e7_arch}_transductive_seed{primary_seed}"
    e7_ind_seed_dir = Path(args.out_root) / "E7" / f"{e7_arch}_inductive_seed{primary_seed}"
    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E7_infer_transductive",
            stage="E7",
            model=f"{e7_arch.upper()}-trans-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-gnn-all",
                "--gnn_data_dir",
                args.gnn_data_dir,
                "--demand_csv",
                args.demand_csv,
                "--base_exp_dir",
                str(e7_trans_seed_dir),
                "--archs",
                e7_arch,
                "--p_add",
                "100",
                "--prob_thresh",
                "0.7",
            ],
            out_dir=str(e7_trans_seed_dir),
            result_type="inference",
            result_file=str(e7_trans_seed_dir / f"inference_results_{e7_arch}.csv"),
        )
    )
    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E7_infer_inductive",
            stage="E7",
            model=f"{e7_arch.upper()}-ind-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-gnn-all",
                "--gnn_data_dir",
                args.gnn_data_dir,
                "--demand_csv",
                args.demand_csv,
                "--base_exp_dir",
                str(e7_ind_seed_dir),
                "--archs",
                e7_arch,
                "--p_add",
                "100",
                "--prob_thresh",
                "0.7",
            ],
            out_dir=str(e7_ind_seed_dir),
            result_type="inference",
            result_file=str(e7_ind_seed_dir / f"inference_results_{e7_arch}.csv"),
        )
    )

    # E8 Fold4 diagnostics (with uncertainty sources from E7).
    e8_out = Path(args.out_root) / "E8_fold4"
    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E8_fold4_analysis",
            stage="E8",
            model="Fold4Diagnostic",
            seed=primary_seed,
            command=[
                py,
                "scripts/run_fold4_analysis.py",
                "--gnn_data_dir",
                args.gnn_data_dir,
                "--out_dir",
                str(e8_out),
                "--folds",
                str(args.folds),
                "--grid_size",
                str(args.grid_size),
                "--seed",
                str(primary_seed),
                "--target_fold",
                "4",
                "--infer_csv",
                f"trans={e7_trans_seed_dir / f'inference_results_{e7_arch}.csv'}",
                "--infer_csv",
                f"ind={e7_ind_seed_dir / f'inference_results_{e7_arch}.csv'}",
            ],
            out_dir=str(e8_out),
            result_type="fold_diag",
            result_file=str(e8_out / "fold_diagnostics.csv"),
        )
    )

    # E9/E10 RMCLP source + budget curves.
    rmclp_out = Path(args.out_root) / "E9E10_rmclp"
    add_rmclp_curve_job(
        jobs=jobs,
        py=py,
        exp_id="E9E10_rmclp_curve",
        stage="E10",
        model="RMCLP-Curve",
        seed=primary_seed,
        out_dir=rmclp_out,
        processed_dir=args.processed_dir,
        demand_csv=args.demand_csv,
        p_add_grid=args.p_add_grid,
        radius_grid=args.radius_grid,
        prob_grid=args.prob_grid,
        selection_modes="rmclp",
        rn_filter_modes="true",
        lambda_disp=args.e15_lambda_disp,
        skip_default_sources=True,
        source_csv=[
            f"cnn={e5_pn_dir / 'inference_results.csv'}",
            f"purn={e5_purn_dir / 'inference_results.csv'}",
            f"purn_gnn={e5_gnn_dir / 'inference_results_sage.csv'}",
        ],
    )

    # E15: RMCLP module ablation (selection rule + RN filtering), fixed source and budget.
    e15_source_path = e5_gnn_dir / "inference_results_sage.csv"
    jobs.append(
        Job(
            job_id=f"job_{len(jobs):04d}",
            exp_id="E15_infer_gnn",
            stage="E15",
            model="PURN+GNN-infer",
            seed=primary_seed,
            command=[
                py,
                "scripts/main.py",
                "infer-gnn-all",
                "--gnn_data_dir",
                args.gnn_data_dir,
                "--demand_csv",
                args.demand_csv,
                "--base_exp_dir",
                str(e5_gnn_dir),
                "--archs",
                "sage",
                "--p_add",
                str(args.e15_p_add),
                "--prob_thresh",
                str(args.e15_prob_thresh),
            ],
            out_dir=str(e5_gnn_dir),
            result_type="inference",
            result_file=str(e15_source_path),
        )
    )
    for selection_mode in ["mclp", "rmclp"]:
        for rn_filter_tag, rn_filter_modes in [("rn_off", "false"), ("rn_on", "true")]:
            add_rmclp_curve_job(
                jobs=jobs,
                py=py,
                exp_id=f"E15_{selection_mode}_{rn_filter_tag}",
                stage="E15",
                model=f"{selection_mode.upper()}-{rn_filter_tag}",
                seed=primary_seed,
                out_dir=Path(args.out_root) / "E15" / f"{selection_mode}_{rn_filter_tag}",
                processed_dir=args.processed_dir,
                demand_csv=args.demand_csv,
                p_add_grid=str(args.e15_p_add),
                radius_grid=str(args.e15_radius_km),
                prob_grid=str(args.e15_prob_thresh),
                selection_modes=selection_mode,
                rn_filter_modes=rn_filter_modes,
                lambda_disp=args.e15_lambda_disp,
                skip_default_sources=True,
                source_csv=[f"{args.e15_source}={e15_source_path}"],
            )

    filtered = [j for j in jobs if should_keep(j, tokens)]
    # Re-number ids after filtering for deterministic logs.
    for idx, job in enumerate(filtered):
        job.job_id = f"job_{idx:04d}"
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--gnn_data_dir", default="data/gnn_ready")
    parser.add_argument("--demand_csv", default="data/processed/demand_per_patch.csv")
    parser.add_argument("--out_root", default="outputs/exp_closure_20260315")

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--subset", nargs="*", default=[])
    parser.add_argument("--fast_2fold", action="store_true")

    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--external_seeds", default="42,52,62")
    parser.add_argument("--e7_archs", default="sage")
    parser.add_argument("--e14_archs", default="mlp,sage")

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--buffer_cells", type=int, default=0, help="Global buffered split radius for non-E7 jobs")
    parser.add_argument("--e7_buffer_cells", type=int, default=0, help="Buffered split radius for E7 buffered protocol")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--cnn_batch_size", type=int, default=128)
    parser.add_argument("--cnn_warm_epochs", type=int, default=5)
    parser.add_argument("--cnn_nnpu_epochs", type=int, default=15)
    parser.add_argument("--cnn_lr", type=float, default=1e-4)

    parser.add_argument("--gnn_epochs", type=int, default=200)
    parser.add_argument("--gnn_lr", type=float, default=0.005)
    parser.add_argument("--upu_epochs", type=int, default=120)
    parser.add_argument("--appnp_epochs", type=int, default=150)
    parser.add_argument("--lp_max_iter", type=int, default=1000)

    parser.add_argument("--rn_slope_default", type=float, default=45.0)
    parser.add_argument("--rn_water_default", type=float, default=0.5)
    parser.add_argument("--lambda_default", type=float, default=1.0)
    parser.add_argument("--pi_ref", type=float, default=0.15)

    parser.add_argument("--e2_slope_grid", default="35,40,45,50")
    parser.add_argument("--e2_water_grid", default="0.3,0.5,0.7")
    parser.add_argument("--e4_lambda_grid", default="0,0.25,0.5,1,2")

    parser.add_argument("--p_add_grid", default="20,50,100,150")
    parser.add_argument("--radius_grid", default="3,5,8")
    parser.add_argument("--prob_grid", default="0.6,0.7,0.8")
    parser.add_argument("--e15_source", default="purn_gnn")
    parser.add_argument("--e15_p_add", type=int, default=100)
    parser.add_argument("--e15_radius_km", type=float, default=5.0)
    parser.add_argument("--e15_prob_thresh", type=float, default=0.7)
    parser.add_argument("--e15_lambda_disp", type=float, default=0.3)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    subset_tokens = parse_name_list(args.subset)
    run_label = build_run_label(",".join(subset_tokens))
    run_started_at = datetime.now().isoformat()
    log_dir = out_root / f"logs_{run_label}"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    if not jobs:
        raise RuntimeError("No jobs selected. Check --subset filter.")

    rows = []
    for j in jobs:
        rows.append(
            {
                "job_id": j.job_id,
                "exp_id": j.exp_id,
                "stage": j.stage,
                "model": j.model,
                "seed": j.seed,
                "command": list2cmdline(j.command),
                "out_dir": j.out_dir,
                "result_type": j.result_type,
                "result_file": j.result_file,
                "negative_file": j.negative_file,
                "status": "planned",
                "return_code": "",
                "runtime_min": "",
                "log_file": str(log_dir / f"{j.job_id}.log"),
                "run_label": run_label,
                "run_started_at": run_started_at,
                "job_key": j.result_file or list2cmdline(j.command),
            }
        )
    df = pd.DataFrame(rows)
    matrix_csv = out_root / "experiment_matrix.csv"
    matrix_csv_snapshot = out_root / f"experiment_matrix_{run_label}.csv"
    cmd_file = out_root / "run_commands.ps1"
    cmd_file_snapshot = out_root / f"run_commands_{run_label}.ps1"
    save_plan_outputs(df, matrix_csv, matrix_csv_snapshot, cmd_file, cmd_file_snapshot)
    print(f"Saved matrix: {matrix_csv}")
    print(f"Saved matrix snapshot: {matrix_csv_snapshot}")
    print(f"Saved command list: {cmd_file}")
    print(f"Saved command snapshot: {cmd_file_snapshot}")
    print(f"Saved logs to: {log_dir}")

    if not args.execute:
        print("Dry run mode. Add --execute to run all jobs.")
        return

    for idx in range(len(df)):
        row = df.iloc[idx]
        job_cmd = row["command"]
        log_file = Path(row["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        cmd = jobs[idx].command
        with log_file.open("w", encoding="utf-8", errors="ignore") as f:
            f.write(f"COMMAND: {job_cmd}\n\n")
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[1]))
        runtime_min = (time.time() - start) / 60.0

        df.loc[idx, "runtime_min"] = runtime_min
        df.loc[idx, "return_code"] = proc.returncode
        if proc.returncode == 0:
            df.loc[idx, "status"] = "success"
        else:
            df.loc[idx, "status"] = "failed"
            if args.stop_on_error:
                save_plan_outputs(df, matrix_csv, matrix_csv_snapshot, cmd_file, cmd_file_snapshot)
                print(f"Job failed: {row['job_id']}. Stopped due to --stop_on_error.")
                return

        save_plan_outputs(df, matrix_csv, matrix_csv_snapshot, cmd_file, cmd_file_snapshot)
        print(f"[{idx + 1}/{len(df)}] {row['job_id']} -> {df.loc[idx, 'status']}")

    print(f"All done. Updated matrix: {matrix_csv}")


if __name__ == "__main__":
    main()
