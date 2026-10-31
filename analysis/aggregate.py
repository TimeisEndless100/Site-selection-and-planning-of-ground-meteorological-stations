# -*- coding: utf-8 -*-
"""结果聚合入口：把 outputs/<out_root>/ 的实验矩阵整理成统计口径统一的表格。

由 scripts/run_closure_aggregate.py 平移而来，行为不变：
- 读 experiment_matrix.csv（缺则从 experiment_matrix_*.csv 快照重建）；
- 按 result_type 装入 cnn/gnn/external/rmclp 观测 → master_results.csv；
- 按 (stage, model) 输出 master_summary.csv（mean ± std）；
- 组内配对 Wilcoxon + Holm + Cohen's d + bootstrap CI → stats_results.csv；
- 失败任务 → negative_results.csv；矩阵 → reproducibility_checklist.csv。

统计检验统一走 analysis/statistics.py（单一实现，禁止各自实现一份）；
矩阵合并走 src/core/matrix.py（与 runner/plan.py 共享）。

用法：
    python -m analysis.aggregate --out_root <out_root> [--matrix_csv …] [--n_boot 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.core.exp_logger import ExperimentLogger
from src.core.matrix import merge_matrices
from analysis.statistics import (
    bootstrap_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_wilcoxon,
)


MASTER_COLS = [
    "exp_id",
    "stage",
    "model",
    "fold",
    "seed",
    "auc",
    "auc_pu",  # 候选域 P∪U P-vs-U AUC：跨模型公平比较主口径（2026-08-12）
    "auprc",
    "brier",
    "ece",      # 全域 ECE（sens_pi 校准论证，2026-08-12 补）
    "ece_pu",   # 候选域 ECE（主口径对应校准）
    "sep_prn",
    "recall_k",
    "rmclp_gain",
    "runtime_min",
]

NEGATIVE_COLS = [
    "exp_id",
    "stage",
    "model",
    "fold",
    "seed",
    "reason",
    "log_file",
]

STATS_COLS = [
    "stage",
    "metric",
    "model_a",
    "model_b",
    "delta_mean",
    "effect_size_d",
    "p_value",
    "ci_low",
    "ci_high",
    "wins_a",
    "wins_b",
    "ties",
    "n_pairs",
    "p_adj",
    "signif_0_05",
]

SUMMARY_COLS = [
    "stage",
    "model",
    "n",
    "auc_mean",
    "auc_std",
    "auc_pu_mean",
    "auc_pu_std",
    "auprc_mean",
    "auprc_std",
    "brier_mean",
    "brier_std",
    "ece_mean",
    "ece_std",
    "ece_pu_mean",
    "ece_pu_std",
    "sep_prn_mean",
    "sep_prn_std",
    "recall_k_mean",
    "recall_k_std",
    "rmclp_gain_mean",
    "rmclp_gain_std",
]


def load_or_rebuild_matrix(matrix_csv: Path, out_root: Path) -> pd.DataFrame:
    if matrix_csv.exists():
        return pd.read_csv(matrix_csv)

    snapshots = sorted(out_root.glob("experiment_matrix_*.csv"))
    if not snapshots:
        raise FileNotFoundError(f"Matrix csv not found: {matrix_csv}")

    merged_df: pd.DataFrame | None = None
    for path in snapshots:
        df = pd.read_csv(path)
        merged_df = df if merged_df is None else merge_matrices(merged_df, df)

    assert merged_df is not None
    merged_df.to_csv(matrix_csv, index=False)
    return merged_df


def load_cnn_rows(meta: pd.Series, result_file: Path) -> list[dict]:
    if not result_file.exists():
        return []
    df = pd.read_csv(result_file)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "exp_id": meta["exp_id"],
                "stage": meta["stage"],
                "model": meta["model"],
                "fold": int(r.get("fold", -1)),
                "seed": int(r.get("seed", meta["seed"] if pd.notna(meta["seed"]) else -1)),
                "auc": float(r["auc"]) if pd.notna(r.get("auc", np.nan)) else np.nan,
                "auc_pu": float(r["auc_pu"]) if pd.notna(r.get("auc_pu", np.nan)) else np.nan,
                "auprc": float(r["auprc"]) if pd.notna(r.get("auprc", np.nan)) else float(r.get("ap", np.nan)),
                "brier": float(r["brier"]) if pd.notna(r.get("brier", np.nan)) else np.nan,
                "ece": float(r["ece"]) if pd.notna(r.get("ece", np.nan)) else np.nan,
                "ece_pu": float(r["ece_pu"]) if pd.notna(r.get("ece_pu", np.nan)) else np.nan,
                "sep_prn": float(r["sep_prn"]) if pd.notna(r.get("sep_prn", np.nan)) else float(r.get("separation", np.nan)),
                "recall_k": float(r["recall_k"]) if pd.notna(r.get("recall_k", np.nan)) else np.nan,
                "rmclp_gain": np.nan,
                "runtime_min": float(r["runtime_min"]) if pd.notna(r.get("runtime_min", np.nan)) else np.nan,
            }
        )
    return rows


def load_gnn_rows(meta: pd.Series, result_file: Path) -> list[dict]:
    if not result_file.exists():
        return []
    df = pd.read_csv(result_file)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "exp_id": meta["exp_id"],
                "stage": meta["stage"],
                "model": meta["model"],
                "fold": int(r.get("fold", -1)),
                "seed": int(r.get("seed", meta["seed"] if pd.notna(meta["seed"]) else -1)),
                "auc": float(r["auc"]) if pd.notna(r.get("auc", np.nan)) else np.nan,
                "auc_pu": float(r["auc_pu"]) if pd.notna(r.get("auc_pu", np.nan)) else np.nan,
                "auprc": float(r["auprc"]) if pd.notna(r.get("auprc", np.nan)) else float(r.get("ap", np.nan)),
                "brier": float(r["brier"]) if pd.notna(r.get("brier", np.nan)) else np.nan,
                "ece": float(r["ece"]) if pd.notna(r.get("ece", np.nan)) else np.nan,
                "ece_pu": float(r["ece_pu"]) if pd.notna(r.get("ece_pu", np.nan)) else np.nan,
                "sep_prn": float(r["sep_prn"]) if pd.notna(r.get("sep_prn", np.nan)) else float(r.get("separation", np.nan)),
                "recall_k": float(r["recall_k"]) if pd.notna(r.get("recall_k", np.nan)) else np.nan,
                "rmclp_gain": np.nan,
                "runtime_min": float(r["runtime_min"]) if pd.notna(r.get("runtime_min", np.nan)) else np.nan,
            }
        )
    return rows


def load_external_rows(result_file: Path) -> list[dict]:
    if not result_file.exists():
        return []
    df = pd.read_csv(result_file)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "exp_id": r.get("exp_id", "E6_external"),
                "stage": r.get("stage", "E6"),
                "model": r.get("model", "external"),
                "fold": int(r.get("fold", -1)),
                "seed": int(r.get("seed", -1)),
                "auc": float(r.get("auc", np.nan)),
                "auc_pu": float(r.get("auc_pu", np.nan)),
                "auprc": float(r.get("auprc", np.nan)),
                "brier": float(r.get("brier", np.nan)),
                "ece": float(r.get("ece", np.nan)),
                "ece_pu": float(r.get("ece_pu", np.nan)),
                "sep_prn": float(r.get("sep_prn", np.nan)),
                "recall_k": float(r.get("recall_k", np.nan)),
                "rmclp_gain": float(r.get("rmclp_gain", np.nan)),
                "runtime_min": float(r.get("runtime_min", np.nan)),
            }
        )
    return rows


def load_rmclp_rows(meta: pd.Series, result_file: Path) -> list[dict]:
    if not result_file.exists():
        return []
    df = pd.read_csv(result_file)
    rows = []
    seed = int(meta["seed"]) if pd.notna(meta["seed"]) else -1
    for _, r in df.iterrows():
        source = str(r.get("candidate_source", "unknown"))
        selection_mode = str(r.get("selection_mode", "") or "").strip()
        rn_filter_val = r.get("rn_filter", np.nan)
        model_parts = [f"RMCLP::{source}"]
        if selection_mode:
            model_parts.append(selection_mode)
        if pd.notna(rn_filter_val):
            model_parts.append("rn_on" if bool(rn_filter_val) else "rn_off")
        rows.append(
            {
                "exp_id": meta["exp_id"],
                "stage": meta["stage"],
                "model": "::".join(model_parts),
                "fold": -1,
                "seed": seed,
                "auc": np.nan,
                "auc_pu": np.nan,
                "auprc": np.nan,
                "brier": np.nan,
                "ece": np.nan,
                "ece_pu": np.nan,
                "sep_prn": np.nan,
                "recall_k": np.nan,
                "rmclp_gain": float(r.get("gain_pct", np.nan)),
                "runtime_min": np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="outputs/exp_closure_20260315")
    parser.add_argument("--matrix_csv", default="")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_root>/logs_aggregate")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    # Keep aggregate logs separate from closure-plan job logs.
    log_dir = Path(args.log_dir) if args.log_dir else out_root / "logs_aggregate"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_root.name,
        stage="closure_aggregate",
        model="aggregate",
        seed=args.seed,
        run_params=vars(args),
        data_paths={"out_root": str(out_root)},
    )
    matrix_csv = Path(args.matrix_csv).resolve() if args.matrix_csv else out_root / "experiment_matrix.csv"
    matrix = load_or_rebuild_matrix(matrix_csv, out_root)
    # Keep matrix as reproducibility checklist regardless of run status.
    reproducibility_path = out_root / "reproducibility_checklist.csv"
    matrix.to_csv(reproducibility_path, index=False)

    master_rows: list[dict] = []
    negative_rows: list[dict] = []

    for _, m in matrix.iterrows():
        result_type = str(m.get("result_type", ""))
        result_file = Path(str(m.get("result_file", "")))
        status = str(m.get("status", "planned"))
        if status == "failed":
            negative_rows.append(
                {
                    "exp_id": m.get("exp_id", ""),
                    "stage": m.get("stage", ""),
                    "model": m.get("model", ""),
                    "fold": -1,
                    "seed": int(m.get("seed", -1)) if pd.notna(m.get("seed", np.nan)) else -1,
                    "reason": f"job_failed(return_code={m.get('return_code', '')})",
                    "log_file": m.get("log_file", ""),
                }
            )

        if result_type == "cnn_results":
            master_rows.extend(load_cnn_rows(m, result_file))
        elif result_type == "gnn_results":
            master_rows.extend(load_gnn_rows(m, result_file))
        elif result_type == "external_results":
            master_rows.extend(load_external_rows(result_file))
            neg_file = str(m.get("negative_file", ""))
            if neg_file:
                nf = Path(neg_file)
                if nf.exists():
                    try:
                        df_neg = pd.read_csv(nf)
                    except pd.errors.EmptyDataError:
                        # 外部基线全部成功时 negative_external.csv 为空文件（仅换行）。
                        # 视为无失败行，不抛异常。
                        df_neg = pd.DataFrame()
                    for _, r in df_neg.iterrows():
                        negative_rows.append(
                            {
                                "exp_id": r.get("exp_id", m.get("exp_id", "")),
                                "stage": r.get("stage", m.get("stage", "")),
                                "model": r.get("model", ""),
                                "fold": int(r.get("fold", -1)),
                                "seed": int(r.get("seed", -1)),
                                "reason": r.get("reason", "external_failure"),
                                "log_file": m.get("log_file", ""),
                            }
                        )
        elif result_type == "rmclp_curve":
            master_rows.extend(load_rmclp_rows(m, result_file))

    master = pd.DataFrame(master_rows)
    if len(master) == 0:
        master = pd.DataFrame(columns=MASTER_COLS)
    else:
        master = master[MASTER_COLS]

    master_path = out_root / "master_results.csv"
    master.to_csv(master_path, index=False)

    # Summary mean ± std.
    summary_rows = []
    if len(master) > 0:
        metric_cols = ["auc", "auc_pu", "auprc", "brier", "ece", "ece_pu", "sep_prn", "recall_k", "rmclp_gain"]
        grp = master.groupby(["stage", "model"], dropna=False)
        for (stage, model), g in grp:
            row = {"stage": stage, "model": model, "n": int(len(g))}
            for c in metric_cols:
                vals = g[c].dropna().values.astype(float)
                row[f"{c}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
                row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLS)
    summary_df.to_csv(out_root / "master_summary.csv", index=False)

    # Pairwise statistical significance (paired by fold+seed within stage).
    stats_rows: list[dict] = []
    metrics = ["auc", "auc_pu", "auprc", "brier", "ece", "ece_pu", "sep_prn", "recall_k", "rmclp_gain"]
    if len(master) > 0:
        for stage, stage_df in master.groupby("stage"):
            models = sorted(stage_df["model"].dropna().unique().tolist())
            for metric in metrics:
                pair_rows = []
                for model_a, model_b in itertools.combinations(models, 2):
                    a = stage_df[stage_df["model"] == model_a][["fold", "seed", metric]].rename(columns={metric: "a"})
                    b = stage_df[stage_df["model"] == model_b][["fold", "seed", metric]].rename(columns={metric: "b"})
                    pair = a.merge(b, on=["fold", "seed"], how="inner").dropna()
                    if len(pair) < 2:
                        continue

                    diffs = (pair["a"] - pair["b"]).values.astype(float)
                    delta_mean = float(np.mean(diffs))
                    p_val = paired_wilcoxon(pair["a"], pair["b"])
                    ci_low, ci_high = bootstrap_ci(diffs, np.mean, n_boot=args.n_boot, seed=args.seed)
                    pair_rows.append(
                        {
                            "stage": stage,
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "delta_mean": delta_mean,
                            "effect_size_d": cohens_d_paired(pair["a"], pair["b"]),
                            "p_value": p_val,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "wins_a": int((diffs > 0).sum()),
                            "wins_b": int((diffs < 0).sum()),
                            "ties": int((diffs == 0).sum()),
                            "n_pairs": int(len(pair)),
                        }
                    )

                if pair_rows:
                    pvals = [r["p_value"] for r in pair_rows if pd.notna(r["p_value"])]
                    adj_vals = holm_bonferroni([float(x) for x in pvals]) if pvals else []
                    idx_adj = 0
                    for r in pair_rows:
                        if pd.notna(r["p_value"]):
                            r["p_adj"] = adj_vals[idx_adj]
                            r["signif_0_05"] = bool(adj_vals[idx_adj] < 0.05)
                            idx_adj += 1
                        else:
                            r["p_adj"] = np.nan
                            r["signif_0_05"] = False
                    stats_rows.extend(pair_rows)

    stats_df = pd.DataFrame(stats_rows, columns=STATS_COLS)
    stats_path = out_root / "stats_results.csv"
    stats_df.to_csv(stats_path, index=False)

    negative_df = pd.DataFrame(negative_rows, columns=NEGATIVE_COLS)
    negative_path = out_root / "negative_results.csv"
    negative_df.to_csv(negative_path, index=False)

    for p, t in [
        (master_path, "master_results_csv"),
        (stats_path, "stats_results_csv"),
        (negative_path, "negative_results_csv"),
        (reproducibility_path, "reproducibility_checklist_csv"),
    ]:
        logger.log_artifact(path=p, artifact_type=t, fold=-1)

    # Consolidated plot-ready artifact index.
    plot_ready_rows: list[dict] = []
    for artifact_csv in sorted(out_root.rglob("artifact_index.csv")):
        try:
            df_art = pd.read_csv(artifact_csv)
        except Exception:
            continue
        if len(df_art) == 0:
            continue
        df_art = df_art.copy()
        df_art["artifact_index_path"] = str(artifact_csv)
        plot_ready_rows.extend(df_art.to_dict("records"))
    plot_ready_path = out_root / "plot_ready_index.csv"
    pd.DataFrame(plot_ready_rows).to_csv(plot_ready_path, index=False)
    logger.log_artifact(path=plot_ready_path, artifact_type="plot_ready_index_csv", fold=-1)
    logger.log_fold_summary(
        fold=-1,
        auc=np.nan,
        auprc=np.nan,
        brier=np.nan,
        sep_prn=np.nan,
        recall_k=np.nan,
        runtime_min=np.nan,
    )

    print(f"Saved: {master_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {negative_path}")
    print(f"Saved: {reproducibility_path}")
    print(f"Saved: {plot_ready_path}")
    logger.finalize_run(status="success")


if __name__ == "__main__":
    main()
