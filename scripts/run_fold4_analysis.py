# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.rn import assign_sets
from src.spatial_split import make_spatial_folds, make_spatial_folds_buffered


def parse_named_paths(items: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def degree_vector(edge_index: torch.Tensor, n_nodes: int) -> np.ndarray:
    edge_np = edge_index.cpu().numpy()
    deg = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(deg, edge_np[0], 1)
    return deg


def internal_edge_density(edge_index: torch.Tensor, node_ids: np.ndarray) -> float:
    node_set = set(int(x) for x in node_ids.tolist())
    edge_np = edge_index.cpu().numpy()
    m = 0
    for u, v in zip(edge_np[0], edge_np[1]):
        if int(u) in node_set and int(v) in node_set:
            m += 1
    n = len(node_ids)
    if n <= 1:
        return 0.0
    # Directed edge index is expected; normalize by n*(n-1)
    return float(m / (n * (n - 1)))


def feature_shift(train_x: np.ndarray, val_x: np.ndarray) -> tuple[float, float]:
    train_mean = train_x.mean(axis=0)
    val_mean = val_x.mean(axis=0)
    shift_l1 = float(np.mean(np.abs(train_mean - val_mean)))
    wd = [wasserstein_distance(train_x[:, i], val_x[:, i]) for i in range(train_x.shape[1])]
    wd_mean = float(np.mean(wd))
    return shift_l1, wd_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_data_dir", default="data/gnn_ready")
    parser.add_argument("--out_dir", default="outputs/exp_closure")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--buffer_cells", type=int, default=0, help="Optional buffered split radius in grid cells")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_fold", type=int, default=4)
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument(
        "--infer_csv",
        action="append",
        help="Optional model uncertainty source in format name=path/to/inference.csv",
    )
    args = parser.parse_args()

    gnn_dir = Path(args.gnn_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.load(gnn_dir / "node_features.pt", map_location="cpu").numpy()
    edge_index = torch.load(gnn_dir / "edge_index.pt", map_location="cpu")
    df = pd.read_csv(gnn_dir / "nodes_metadata.csv")
    df = assign_sets(
        df,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy="physics",
        seed=args.seed,
    )

    if args.buffer_cells > 0:
        folds = make_spatial_folds_buffered(
            df,
            n_splits=args.folds,
            grid_size=args.grid_size,
            seed=args.seed,
            buffer_cells=args.buffer_cells,
        )
    else:
        folds = make_spatial_folds(df, n_splits=args.folds, grid_size=args.grid_size, seed=args.seed)
    deg = degree_vector(edge_index=edge_index, n_nodes=len(df))

    infer_sources = parse_named_paths(args.infer_csv)
    infer_tables: dict[str, pd.DataFrame] = {}
    for name, path in infer_sources.items():
        if not path.exists():
            print(f"[WARN] Inference csv not found, skip source '{name}': {path}")
            continue
        infer_tables[name] = pd.read_csv(path)

    rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)

        train_x = x[train_idx]
        val_x = x[val_idx]
        shift_l1, wd_mean = feature_shift(train_x, val_x)

        val_deg = deg[val_idx]
        val_density = internal_edge_density(edge_index=edge_index, node_ids=val_idx)

        row = {
            "fold": fold_idx,
            "val_nodes": int(len(val_idx)),
            "val_degree_mean": float(val_deg.mean()),
            "val_degree_p25": float(np.percentile(val_deg, 25)),
            "val_degree_p50": float(np.percentile(val_deg, 50)),
            "val_degree_p75": float(np.percentile(val_deg, 75)),
            "val_internal_density": val_density,
            "feature_shift_l1": shift_l1,
            "feature_shift_wd_mean": wd_mean,
        }

        fold_df = df.iloc[val_idx].copy()
        row["val_pos_rate"] = float((fold_df["set"] == "P").mean())
        row["val_rn_rate"] = float((fold_df["set"] == "RN").mean())

        for name, infer_df in infer_tables.items():
            pid_set = set(fold_df["pid"].values.tolist())
            sub = infer_df[infer_df["pid"].isin(pid_set)]
            if len(sub) == 0:
                row[f"{name}_prob_mean"] = np.nan
                row[f"{name}_uncertainty_mean"] = np.nan
            else:
                row[f"{name}_prob_mean"] = float(sub["prob_mean"].mean())
                if "prob_std" in sub.columns:
                    row[f"{name}_uncertainty_mean"] = float(sub["prob_std"].mean())
                else:
                    row[f"{name}_uncertainty_mean"] = np.nan

        rows.append(row)

    out_csv = out_dir / "fold_diagnostics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    df_diag = pd.DataFrame(rows)
    target = df_diag[df_diag["fold"] == args.target_fold]
    if len(target) == 0:
        raise ValueError(f"target_fold={args.target_fold} is not present in diagnostics table.")
    target_row = target.iloc[0].to_dict()
    others = df_diag[df_diag["fold"] != args.target_fold]
    summary = {
        "target_fold": args.target_fold,
        "target": target_row,
        "others_mean": others.mean(numeric_only=True).to_dict(),
        "others_std": others.std(numeric_only=True).to_dict(),
    }
    out_json = out_dir / "fold4_diagnostics_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
