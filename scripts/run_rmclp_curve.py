# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.exp_logger import ExperimentLogger, build_inference_summary
from src.core.rmclp import rmclp_gain
from src.core.seed import set_seed


def ensure_purn_infer(processed_dir: Path, exp_dir: Path) -> Path | None:
    infer_path = exp_dir / "inference_results.csv"
    if infer_path.exists():
        return infer_path
    if not list(exp_dir.glob("fold_*/model_best.pth")):
        return None
    cmd = [
        "python",
        "scripts/run_infer.py",
        "--processed_dir",
        str(processed_dir),
        "--exp_dir",
        str(exp_dir),
        "--out_name",
        "inference_results.csv",
    ]
    subprocess.run(cmd, check=True)
    return infer_path if infer_path.exists() else None


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


def main() -> None:
    parser = argparse.ArgumentParser()
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
        sources["random"] = build_random_source(df_base, seed=args.seed)
        sources["gis_rule"] = build_gis_source(df_base)
        cnn_path = Path("outputs/exp_pu_no_coor/inference_results.csv")
        if cnn_path.exists():
            sources["cnn"] = pd.read_csv(cnn_path)
        purn_path = ensure_purn_infer(processed_dir, Path("outputs/exp_pu_final"))
        if purn_path is not None and purn_path.exists():
            sources["purn"] = pd.read_csv(purn_path)
        gnn_path = Path("outputs/exp_gnn_compare/inference_results_sage.csv")
        if gnn_path.exists():
            sources["purn_gnn"] = pd.read_csv(gnn_path)

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


if __name__ == "__main__":
    main()
