# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from src.gnn_model import SpatialGAT, SpatialGCN, SpatialSAGE
except ImportError as ex:
    raise ImportError(
        "torch_geometric is required for run_infer_gnn_all.py. "
        "Install dependencies from requirements.txt."
    ) from ex

from src.core.exp_logger import ExperimentLogger, build_inference_summary
from src.core.rmclp import rmclp_select_sites


def build_model(arch: str, in_channels: int) -> torch.nn.Module:
    if arch == "gat":
        return SpatialGAT(in_channels=in_channels, hidden_channels=64, out_channels=1)
    if arch == "gcn":
        return SpatialGCN(in_channels=in_channels, hidden_channels=64, out_channels=1)
    if arch == "sage":
        return SpatialSAGE(in_channels=in_channels, hidden_channels=64, out_channels=1)
    raise ValueError(f"Unknown architecture: {arch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_data_dir", default="data/gnn_ready")
    parser.add_argument("--demand_csv", default="data/processed/demand_per_patch.csv")
    parser.add_argument("--base_exp_dir", default="outputs/exp_gnn_compare")
    parser.add_argument("--archs", nargs="+", default=["sage", "gat", "gcn"])
    parser.add_argument("--p_add", type=int, default=100)
    parser.add_argument("--prob_thresh", type=float, default=0.7)
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <base_exp_dir>/logs_infer_gnn_all")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    base_exp_dir = Path(args.base_exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep inference/selection logs isolated from per-architecture training logs.
    log_dir = Path(args.log_dir) if args.log_dir else base_exp_dir / "logs_infer_gnn_all"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=base_exp_dir.name,
        stage="infer_gnn_all",
        model="multi_arch",
        seed=-1,
        run_params=vars(args),
        data_paths={"gnn_data_dir": args.gnn_data_dir, "demand_csv": args.demand_csv},
    )

    feature_path = Path(args.gnn_data_dir) / "node_features.pt"
    edge_path = Path(args.gnn_data_dir) / "edge_index.pt"
    meta_path = Path(args.gnn_data_dir) / "nodes_metadata.csv"
    if not (feature_path.exists() and edge_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing data files in {args.gnn_data_dir}")

    x = torch.load(feature_path).to(device)
    edge_index = torch.load(edge_path).to(device)
    df_meta = pd.read_csv(meta_path)
    if not Path(args.demand_csv).exists():
        raise FileNotFoundError(f"Demand CSV not found: {args.demand_csv}")
    df_demand = pd.read_csv(args.demand_csv)

    summary_rows: list[dict] = []
    infer_summary_rows: list[dict] = []
    filter_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for arch in args.archs:
        print(f"\n================ Processing: {arch.upper()} ================")
        model_dir = base_exp_dir / arch
        if not model_dir.exists():
            print(f"Skipping {arch}, directory not found: {model_dir}")
            continue

        model_paths = sorted(model_dir.glob("fold_*/gnn_best.pth"))
        if not model_paths:
            print(f"Skipping {arch}, no model checkpoints found.")
            continue

        all_preds = np.zeros((x.shape[0], len(model_paths)))
        in_channels = int(x.shape[1])

        for i, path in enumerate(model_paths):
            model = build_model(arch, in_channels).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x, edge_index)).cpu().numpy()
            all_preds[:, i] = probs

        df_out = df_meta.copy()
        df_out["prob_mean"] = np.mean(all_preds, axis=1)
        df_out["prob_std"] = np.std(all_preds, axis=1)

        infer_csv_path = base_exp_dir / f"inference_results_{arch}.csv"
        df_out.to_csv(infer_csv_path, index=False)
        logger.log_artifact(path=infer_csv_path, artifact_type=f"inference_results_{arch}", fold=-1)
        print(f"Inference map saved to: {infer_csv_path}")

        infer_row = build_inference_summary(df_out)
        infer_row["arch"] = arch
        infer_summary_rows.append(infer_row)

        result = rmclp_select_sites(
            infer_df=df_out,
            demand_df=df_demand,
            p_add=args.p_add,
            prob_thresh=args.prob_thresh,
        )
        df_sel = result["selected_sites"]
        base_cov = float(result["base_cov"])
        final_cov = float(result["final_cov"])
        gain = float(result["gain"])

        if len(df_sel) > 0:
            sel_csv_path = base_exp_dir / f"selected_sites_{arch}.csv"
            df_sel.to_csv(sel_csv_path, index=False)
            logger.log_artifact(path=sel_csv_path, artifact_type=f"selected_sites_{arch}", fold=-1)
            print(f"Selection saved to: {sel_csv_path}")
            print(f"Coverage Improvement: {gain:.2f} (Total: {final_cov:.2f})")
        else:
            print("No valid candidates found for selection.")

        summary_rows.append(
            {
                "Architecture": arch,
                "Base Coverage": base_cov,
                "Final Coverage": final_cov,
                "Gain": gain,
                "Candidates Found": int(result["candidates"]),
            }
        )

        frow = dict(result.get("candidate_filter_stats", {}))
        frow["arch"] = arch
        frow["base_cov"] = base_cov
        frow["final_cov"] = final_cov
        frow["gain"] = gain
        frow["gain_pct"] = float(result["gain_pct"])
        filter_rows.append(frow)
        logger.log_fold_summary(
            fold=-1,
            auc=np.nan,
            auprc=np.nan,
            brier=np.nan,
            sep_prn=float(result["gain_pct"]),
            recall_k=np.nan,
            runtime_min=np.nan,
        )

        for c in result.get("coverage_curve", []):
            row = dict(c)
            row["arch"] = arch
            coverage_rows.append(row)

    if summary_rows:
        res_df = pd.DataFrame(summary_rows)
        summary_path = base_exp_dir / "comparison_summary.csv"
        res_df.to_csv(summary_path, index=False)
        logger.log_artifact(path=summary_path, artifact_type="comparison_summary_csv", fold=-1)
        print("\n================ Final Summary ================")
        print(res_df)

    if infer_summary_rows:
        infer_summary_path = base_exp_dir / "inference_summary.csv"
        pd.DataFrame(infer_summary_rows).to_csv(infer_summary_path, index=False)
        logger.log_artifact(path=infer_summary_path, artifact_type="inference_summary_csv", fold=-1)

    if filter_rows:
        filter_path = base_exp_dir / "candidate_filter_stats.csv"
        pd.DataFrame(filter_rows).to_csv(filter_path, index=False)
        logger.log_artifact(path=filter_path, artifact_type="candidate_filter_stats_csv", fold=-1)

    if coverage_rows:
        coverage_path = base_exp_dir / "rmclp_coverage_curve.csv"
        pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
        logger.log_artifact(path=coverage_path, artifact_type="rmclp_coverage_curve_csv", fold=-1)

    logger.finalize_run(status="success")


if __name__ == "__main__":
    main()
