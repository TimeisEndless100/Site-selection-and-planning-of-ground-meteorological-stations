# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.exp_logger import ExperimentLogger, build_inference_summary
from src.core.rmclp import rmclp_select_sites


def main() -> None:
    parser = argparse.ArgumentParser()
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


if __name__ == "__main__":
    main()
