# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.core.exp_logger import ExperimentLogger, build_inference_summary
from src.dataset import PUSpatialDataset
from src.model import GeoConvNeXt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--exp_dir", default="outputs/exp_v1")
    parser.add_argument("--out_name", default="inference_results.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <exp_dir>/logs_infer_cnn")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keep inference logs isolated from training logs under the same experiment dir.
    log_dir = Path(args.log_dir) if args.log_dir else exp_dir / "logs_infer_cnn"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=exp_dir.name,
        stage="infer_cnn",
        model="cnn",
        seed=-1,
        run_params=vars(args),
        data_paths={"processed_dir": args.processed_dir, "exp_dir": args.exp_dir},
    )

    print("Loading patch info...")
    df = pd.read_csv(Path(args.processed_dir) / "patches_info.csv")
    logger.log_data_split({"fold": -1, "total_nodes": int(len(df))})
    fold_dirs = sorted(exp_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {exp_dir}")

    print(f"Found {len(fold_dirs)} folds. Starting ensemble inference...")
    all_preds = np.zeros((len(df), len(fold_dirs)))
    valid_cols = 0

    for i, fold_dir in enumerate(fold_dirs):
        model_path = fold_dir / "model_best.pth"
        stats_path = fold_dir / "stats.json"
        if not model_path.exists():
            print(f"Skipping {fold_dir.name}, model_best.pth not found.")
            continue
        if not stats_path.exists():
            print(f"Skipping {fold_dir.name}, stats.json not found.")
            continue

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        ds = PUSpatialDataset(df, args.processed_dir, stats)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = GeoConvNeXt(in_ch=ds.input_channels).to(device)
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as ex:
            ckpt_in_ch = None
            stem_w = state_dict.get("stem.0.weight") if isinstance(state_dict, dict) else None
            if stem_w is not None and hasattr(stem_w, "shape") and len(stem_w.shape) >= 2:
                ckpt_in_ch = int(stem_w.shape[1])
            raise RuntimeError(
                "Checkpoint/model channel mismatch during infer-cnn. "
                f"checkpoint_in_ch={ckpt_in_ch}, infer_in_ch={ds.input_channels}. "
                "This usually means the checkpoint was trained on a different input "
                "feature composition than the current processed_dir/stats. "
                "Use the same processed data and stats.json for train and infer."
            ) from ex
        model.eval()

        preds = []
        with torch.no_grad():
            for x, _ in tqdm(dl, desc=f"Infer {fold_dir.name}", leave=False):
                preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        all_preds[:, i] = np.concatenate(preds)
        valid_cols += 1

    if valid_cols == 0:
        raise RuntimeError("No valid fold checkpoint was loaded for inference.")

    all_preds = all_preds[:, :valid_cols]
    df["prob_mean"] = np.mean(all_preds, axis=1)
    df["prob_std"] = np.std(all_preds, axis=1)

    out_path = exp_dir / args.out_name
    cols_to_save = ["pid", "shard", "cell_id", "lon", "lat", "is_P", "set", "prob_mean", "prob_std"]
    if "slope_max" in df.columns:
        cols_to_save.append("slope_max")
    df[cols_to_save].to_csv(out_path, index=False)
    logger.log_artifact(path=out_path, artifact_type="inference_results_csv", fold=-1)
    print(f"Inference done. Saved to {out_path}")

    infer_summary = build_inference_summary(df)
    infer_summary_path = exp_dir / "inference_summary.csv"
    pd.DataFrame([infer_summary]).to_csv(infer_summary_path, index=False)
    logger.log_artifact(path=infer_summary_path, artifact_type="inference_summary_csv", fold=-1)

    candidate_stats = {
        "total_nodes": int(len(df)),
        "prob_ge_0_6": int((df["prob_mean"] >= 0.6).sum()),
        "prob_ge_0_7": int((df["prob_mean"] >= 0.7).sum()),
        "prob_ge_0_8": int((df["prob_mean"] >= 0.8).sum()),
        "prob_ge_0_8_non_existing": int(((df["prob_mean"] >= 0.8) & (df["is_P"] == 0)).sum()),
    }
    candidate_stats_path = exp_dir / "candidate_filter_stats.csv"
    pd.DataFrame([candidate_stats]).to_csv(candidate_stats_path, index=False)
    logger.log_artifact(path=candidate_stats_path, artifact_type="candidate_filter_stats_csv", fold=-1)
    logger.log_fold_summary(
        fold=-1,
        auc=np.nan,
        auprc=np.nan,
        brier=np.nan,
        sep_prn=np.nan,
        recall_k=np.nan,
        runtime_min=np.nan,
    )

    n_high = candidate_stats["prob_ge_0_8_non_existing"]
    print(f"Statistics: Found {n_high} potential sites (Prob>0.8, Not Existing Station)")
    logger.finalize_run(status="success")


if __name__ == "__main__":
    main()
