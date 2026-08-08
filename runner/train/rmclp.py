# -*- coding: utf-8 -*-
"""RMCLP 选址：single（单次选择）与 curve（参数网格扫描）两个入口。

注意：历史工具。论文应用链已移除 RMCLP 决策（app_rmclp_gain / app_rmclp_vs_mclp
已撤销，见 experiments/registry.py），本模块仅供手动演示/留档，不再有 plan 作业。
- single: rmclp_select_sites 单次选择，输出 selected_sites / candidate_filter_stats / coverage_curve。
- curve: 对 p_add × radius × prob × selection_mode × rn_filter 全网格调用 rmclp_gain；
  候选源 = 内置规则源 random/gis_rule + --source_csv 指定（不再硬编码历史输出路径）。

用法：
    python -m runner.train.rmclp single --infer_csv … --demand_csv …
    python -m runner.train.rmclp curve  --out_dir … --source_csv name=path …
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.core.exp_logger import ExperimentLogger, build_inference_summary
from src.core.rmclp import rmclp_gain, rmclp_select_sites
from src.core.seed import set_seed


def build_random_source(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    out = df.copy()
    out["prob_mean"] = rng.rand(len(out))
    out["prob_std"] = 0.0
    return out


def build_gis_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    slope = out["slope_max"].astype(float).values
    water = out["water_ratio"].astype(float).values
    dem = out["dem_mean"].astype(float).values
    slope_term = np.clip(1.0 - (slope / 50.0), 0.0, 1.0)
    water_term = np.clip(1.0 - water, 0.0, 1.0)
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
    dem_term = 1.0 - np.abs(dem_norm - 0.5) * 2.0
    score = 0.5 * slope_term + 0.3 * water_term + 0.2 * np.clip(dem_term, 0.0, 1.0)
    out["prob_mean"] = np.clip(score, 0.0, 1.0)
    out["prob_std"] = 0.0
    return out


def parse_source_csv(items: list[str] | None) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"--source_csv must use name=path format, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --source_csv item: {item}")
        out[name] = Path(path)
    return out


def parse_bool_list(text: str) -> list[bool]:
    mapping = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    values: list[bool] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Invalid boolean token: {token}")
        values.append(mapping[key])
    return values


def main_single() -> None:
    parser = argparse.ArgumentParser(prog="python -m runner.train.rmclp single")
    parser.add_argument("--infer_csv", required=True)
    parser.add_argument("--demand_csv", required=True)
    parser.add_argument("--out_dir", default="outputs/rmclp")
    parser.add_argument("--prob_thresh", type=float, default=0.7)
    parser.add_argument("--rn_filter", action="store_true")
    parser.add_argument("--p_add", type=int, default=100)
    parser.add_argument("--radius_km", type=float, default=5.0)
    parser.add_argument("--selection_mode", type=str, default="rmclp", choices=["mclp", "rmclp"])
    parser.add_argument("--lambda_disp", type=float, default=0.3)
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_dir>/logs")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_dir.name,
        stage="rmclp",
        model="rmclp",
        seed=-1,
        run_params=vars(args),
        data_paths={"infer_csv": args.infer_csv, "demand_csv": args.demand_csv},
    )

    print("Loading data...")
    df_infer = pd.read_csv(args.infer_csv)
    df_demand = pd.read_csv(args.demand_csv)
    logger.log_data_split(
        {
            "fold": -1,
            "infer_nodes": int(len(df_infer)),
            "demand_nodes": int((df_demand["demand"] > 1e-6).sum()) if "demand" in df_demand.columns else int(len(df_demand)),
        }
    )
    infer_summary = build_inference_summary(df_infer)
    infer_summary_path = out_dir / "inference_summary.csv"
    pd.DataFrame([infer_summary]).to_csv(infer_summary_path, index=False)
    logger.log_artifact(path=infer_summary_path, artifact_type="inference_summary_csv", fold=-1)

    result = rmclp_select_sites(
        infer_df=df_infer,
        demand_df=df_demand,
        prob_thresh=args.prob_thresh,
        p_add=args.p_add,
        radius_km=args.radius_km,
        rn_filter=args.rn_filter,
        selection_mode=args.selection_mode,
        lambda_disp=args.lambda_disp,
    )
    final_coverage = float(result["final_cov"])
    base_coverage = float(result["base_cov"])
    df_selected = result["selected_sites"]

    print(f"Candidates (S): {result['candidates']} (from total {len(df_infer)})")
    print(f"Selected Sites: {result['selected']}")
    print(f"Base Coverage (by Existing): {base_coverage:.2f}")
    print(f"Final Coverage: {final_coverage:.2f}")
    print(f"Improvement: {result['gain']:.2f} ({result['gain_pct']:.3f}%)")

    selected_path = out_dir / f"selected_sites_p{args.p_add}.csv"
    df_selected.to_csv(selected_path, index=False)
    logger.log_artifact(path=selected_path, artifact_type="selected_sites_csv", fold=-1)
    print(f"Selected sites saved to {selected_path}")

    filter_stats_path = out_dir / "candidate_filter_stats.csv"
    pd.DataFrame([result.get("candidate_filter_stats", {})]).to_csv(filter_stats_path, index=False)
    logger.log_artifact(path=filter_stats_path, artifact_type="candidate_filter_stats_csv", fold=-1)

    coverage_path = out_dir / "rmclp_coverage_curve.csv"
    pd.DataFrame(result.get("coverage_curve", [])).to_csv(coverage_path, index=False)
    logger.log_artifact(path=coverage_path, artifact_type="rmclp_coverage_curve_csv", fold=-1)
    logger.log_fold_summary(
        fold=-1,
        auc=np.nan,
        auprc=np.nan,
        brier=np.nan,
        sep_prn=float(result["gain_pct"]),
        recall_k=np.nan,
        runtime_min=np.nan,
    )

    logger.finalize_run(status="success")


def main_curve() -> None:
    parser = argparse.ArgumentParser(prog="python -m runner.train.rmclp curve")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--demand_csv", type=str, default="data/processed/demand_per_patch.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/exp_closure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_add_grid", type=str, default="20,50,100,150")
    parser.add_argument("--radius_grid", type=str, default="3,5,8")
    parser.add_argument("--prob_grid", type=str, default="0.6,0.7,0.8")
    parser.add_argument("--selection_modes", type=str, default="rmclp")
    parser.add_argument("--rn_filter_modes", type=str, default="true")
    parser.add_argument("--lambda_disp", type=float, default=0.3)
    parser.add_argument("--source_csv", action="append", help="Custom source: name=path/to/inference.csv")
    parser.add_argument("--skip_default_sources", action="store_true")
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_dir>/logs")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_dir.name,
        stage="rmclp_curve",
        model="rmclp_curve",
        seed=args.seed,
        run_params=vars(args),
        data_paths={"processed_dir": args.processed_dir, "demand_csv": args.demand_csv},
    )

    processed_dir = Path(args.processed_dir)
    df_base = pd.read_csv(processed_dir / "patches_info.csv")
    demand_df = pd.read_csv(args.demand_csv)

    sources: Dict[str, pd.DataFrame] = {}
    if not args.skip_default_sources:
        # 只构建规则源；学得模型源一律经 --source_csv 显式传入（不再硬编码历史输出路径）
        sources["random"] = build_random_source(df_base, seed=args.seed)
        sources["gis_rule"] = build_gis_source(df_base)

    for name, csv_path in parse_source_csv(args.source_csv).items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Source csv not found for {name}: {csv_path}")
        sources[name] = pd.read_csv(csv_path)
    if not sources:
        raise RuntimeError("No candidate source available. Provide --source_csv or keep default sources.")

    p_add_grid = [int(x) for x in args.p_add_grid.split(",") if x.strip()]
    radius_grid = [float(x) for x in args.radius_grid.split(",") if x.strip()]
    prob_grid = [float(x) for x in args.prob_grid.split(",") if x.strip()]
    selection_modes = [x.strip() for x in args.selection_modes.split(",") if x.strip()]
    rn_filter_modes = parse_bool_list(args.rn_filter_modes)

    rmclp_rows: list[dict] = []
    filter_rows: list[dict] = []
    coverage_rows: list[dict] = []
    infer_rows: list[dict] = []

    for source_name, infer_df in sources.items():
        logger.log_data_split({"fold": -1, "candidate_source": source_name, "total_nodes": int(len(infer_df))})
        irow = build_inference_summary(infer_df)
        irow["candidate_source"] = source_name
        infer_rows.append(irow)
        for p_add in p_add_grid:
            for radius in radius_grid:
                for prob in prob_grid:
                    for selection_mode in selection_modes:
                        for rn_filter in rn_filter_modes:
                            metrics = rmclp_gain(
                                infer_df=infer_df,
                                demand_df=demand_df,
                                prob_thresh=prob,
                                p_add=p_add,
                                radius_km=radius,
                                rn_filter=rn_filter,
                                selection_mode=selection_mode,
                                lambda_disp=args.lambda_disp,
                                return_details=True,
                            )
                            rmclp_rows.append(
                                {
                                    "candidate_source": source_name,
                                    "selection_mode": selection_mode,
                                    "rn_filter": bool(rn_filter),
                                    "lambda_disp": float(args.lambda_disp),
                                    "p_add": p_add,
                                    "radius_km": radius,
                                    "prob_thresh": prob,
                                    "base_cov": metrics["base_cov"],
                                    "final_cov": metrics["final_cov"],
                                    "gain": metrics["gain"],
                                    "gain_pct": metrics["gain_pct"],
                                    "candidates": metrics["candidates"],
                                    "selected": metrics["selected"],
                                }
                            )

                            frow = dict(metrics.get("candidate_filter_stats", {}))
                            frow.update(
                                {
                                    "candidate_source": source_name,
                                    "selection_mode": selection_mode,
                                    "rn_filter": bool(rn_filter),
                                    "lambda_disp": float(args.lambda_disp),
                                    "p_add": p_add,
                                    "radius_km": radius,
                                    "prob_thresh": prob,
                                }
                            )
                            filter_rows.append(frow)

                            for row in metrics.get("coverage_curve", []):
                                crow = dict(row)
                                crow.update(
                                    {
                                        "candidate_source": source_name,
                                        "selection_mode": selection_mode,
                                        "rn_filter": bool(rn_filter),
                                        "lambda_disp": float(args.lambda_disp),
                                        "p_add": p_add,
                                        "radius_km": radius,
                                        "prob_thresh": prob,
                                    }
                                )
                                coverage_rows.append(crow)

    rmclp_curve_path = out_dir / "rmclp_curve.csv"
    pd.DataFrame(rmclp_rows).to_csv(rmclp_curve_path, index=False)
    logger.log_artifact(path=rmclp_curve_path, artifact_type="rmclp_curve_csv", fold=-1)

    infer_summary_path = out_dir / "inference_summary.csv"
    pd.DataFrame(infer_rows).to_csv(infer_summary_path, index=False)
    logger.log_artifact(path=infer_summary_path, artifact_type="inference_summary_csv", fold=-1)

    filter_stats_path = out_dir / "candidate_filter_stats.csv"
    pd.DataFrame(filter_rows).to_csv(filter_stats_path, index=False)
    logger.log_artifact(path=filter_stats_path, artifact_type="candidate_filter_stats_csv", fold=-1)

    coverage_path = out_dir / "rmclp_coverage_curve.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    logger.log_artifact(path=coverage_path, artifact_type="rmclp_coverage_curve_csv", fold=-1)
    logger.log_fold_summary(
        fold=-1,
        auc=np.nan,
        auprc=np.nan,
        brier=np.nan,
        sep_prn=float(np.nanmean([r.get("gain_pct", np.nan) for r in rmclp_rows])) if len(rmclp_rows) > 0 else np.nan,
        recall_k=np.nan,
        runtime_min=np.nan,
    )

    print(f"Saved RMCLP curve table: {rmclp_curve_path}")
    logger.finalize_run(status="success")


def main() -> None:
    # 首个参数为子命令名（single / curve），剥去后委托给对应 main_*（各自用 argparse 全量解析）。
    if len(sys.argv) < 2 or sys.argv[1] not in ("single", "curve"):
        print(__doc__)
        if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
            raise SystemExit(0)  # 顶层 --help：打印用法，正常退出
        raise SystemExit("用法: python -m runner.train.rmclp <single|curve> [args…]")
    cmd = sys.argv.pop(1)
    if cmd == "single":
        main_single()
    else:
        main_curve()


if __name__ == "__main__":
    main()
