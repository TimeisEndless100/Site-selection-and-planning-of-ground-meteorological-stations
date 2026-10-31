# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import APPNP
except ImportError as ex:
    raise ImportError(
        "torch_geometric is required for run_external_baselines.py. "
        "Install dependencies from requirements.txt."
    ) from ex

from src.spatial_split import make_spatial_folds, make_spatial_folds_buffered
from src.loss import nnpu_loss
from src.core.exp_logger import ExperimentLogger
from src.core.rn import assign_sets
from src.core.metrics import compute_metric_pack
from src.core.seed import set_seed
from runner.train.extract_features import extract_fold_features, load_pid_sorted_df


class LinearUPU(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)


class APPNPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.prop = APPNP(K=10, alpha=0.1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x).squeeze(1)
        x = self.prop(x.unsqueeze(1), edge_index).squeeze(1)
        return x


def train_upu_predict(
    x_train: np.ndarray,
    train_sets: np.ndarray,
    x_val: np.ndarray,
    pi: float,
    epochs: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    model = LinearUPU(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    sets = torch.tensor(train_sets, dtype=torch.long, device=device)

    p_mask = sets == 1
    u_mask = sets != 1
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(xtr)
        loss = nnpu_loss(logits[p_mask], logits[u_mask], pi=pi)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xva)).cpu().numpy()
    return probs


def train_elkan_noto_predict(
    x_train: np.ndarray,
    y_train_binary: np.ndarray,
    x_val: np.ndarray,
    pos_mask_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=seed)
    clf.fit(x_train, y_train_binary)
    g_train_pos = clf.predict_proba(x_train[pos_mask_train])[:, 1]
    c = float(np.clip(g_train_pos.mean(), 1e-3, 1.0))
    g_val = clf.predict_proba(x_val)[:, 1]
    return np.clip(g_val / c, 0.0, 1.0)


def train_bagging_pu_predict(
    x_train: np.ndarray,
    pos_mask_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
    bags: int = 10,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pos_idx = np.where(pos_mask_train)[0]
    unl_idx = np.where(~pos_mask_train)[0]
    if len(pos_idx) == 0 or len(unl_idx) == 0:
        return np.zeros(len(x_val), dtype=float)

    probs_acc = np.zeros(len(x_val), dtype=float)
    for b in range(bags):
        sample_unl = rng.choice(unl_idx, size=min(len(pos_idx), len(unl_idx)), replace=len(unl_idx) < len(pos_idx))
        train_idx = np.concatenate([pos_idx, sample_unl])
        y = np.zeros(len(train_idx), dtype=int)
        y[: len(pos_idx)] = 1
        clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed + b)
        clf.fit(x_train[train_idx], y)
        probs_acc += clf.predict_proba(x_val)[:, 1]
    return probs_acc / bags


def train_label_prop_predict(
    x_all: np.ndarray,
    train_idx: np.ndarray,
    train_sets: np.ndarray,
    val_idx: np.ndarray,
    max_iter: int = 1000,
) -> tuple[np.ndarray, bool, int]:
    y = np.full(x_all.shape[0], -1, dtype=int)
    tr_is_pos = train_sets == 1
    tr_is_rn = train_sets == 2
    y[train_idx[tr_is_pos]] = 1
    y[train_idx[tr_is_rn]] = 0

    lp = LabelPropagation(kernel="knn", n_neighbors=15, max_iter=max_iter)
    lp.fit(x_all, y)
    prob_pos = lp.label_distributions_[:, 1]
    n_iter = int(getattr(lp, "n_iter_", max_iter))
    converged = n_iter < max_iter
    return prob_pos[val_idx], converged, n_iter


def train_appnp_predict(
    x_all: np.ndarray,
    edge_index: torch.Tensor,
    train_idx: np.ndarray,
    train_sets: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int = 150,
    lr: float = 0.005,
) -> np.ndarray:
    set_seed(seed)
    x_t = torch.tensor(x_all, dtype=torch.float32)
    y = torch.zeros(x_t.size(0), dtype=torch.float32)
    y[torch.tensor(train_idx[train_sets == 1], dtype=torch.long)] = 1.0

    train_sup_mask = torch.zeros(x_t.size(0), dtype=torch.bool)
    train_sup_mask[torch.tensor(train_idx[(train_sets == 1) | (train_sets == 2)], dtype=torch.long)] = True
    val_mask = torch.zeros(x_t.size(0), dtype=torch.bool)
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True

    data = Data(x=x_t, edge_index=edge_index, y=y)
    data = data.to(device)
    train_sup_mask = train_sup_mask.to(device)
    val_mask = val_mask.to(device)

    model = APPNPBaseline(in_dim=x_all.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_sup_mask], data.y[train_sup_mask])
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index)).detach().cpu().numpy()
    return probs[val_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_data_dir", type=str, default="data/gnn_ready")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="patches_info.csv 所在目录（自动提取折特征时的元数据源）")
    parser.add_argument("--cnn_ckpt_dir", type=str, default="",
                        help="PURN CNN 主配置训练目录（含 fold_*/model_best.pth + stats.json）；"
                             "折特征文件缺失时用于自动提取（P0-1）")
    parser.add_argument("--out_dir", type=str, default="outputs/exp_closure/external")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--buffer_cells", type=int, default=0, help="Optional buffered split radius in grid cells")
    parser.add_argument("--seeds", type=str, default="42")  # 2026-08-12 单 seed 决策
    parser.add_argument("--fold_ids", type=str, default="")
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument("--upu_epochs", type=int, default=120)
    parser.add_argument("--appnp_epochs", type=int, default=150)
    parser.add_argument("--lp_max_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
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
        stage="external_baselines",
        model="external",
        seed=-1,
        run_params=vars(args),
        data_paths={"gnn_data_dir": args.gnn_data_dir},
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    # P0-1（2026-08-12）：不再读全局 node_features.pt。特征在折循环内按折加载
    # （node_features_fold_<f>.pt），与 GNN 同源 → 图域基线折对齐公平。
    gnn_dir = Path(args.gnn_data_dir)
    meta_path = gnn_dir / "nodes_metadata.csv"
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
    else:
        df_meta = load_pid_sorted_df(args.processed_dir)
        df_meta.to_csv(meta_path, index=False)
    edge_index = torch.load(gnn_dir / "edge_index.pt", map_location="cpu")
    df = assign_sets(
        df_meta,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy="physics",
        seed=42,
    ).reset_index(drop=True)

    all_rows = []
    negative_rows = []
    for seed in seeds:
        set_seed(seed)
        if args.buffer_cells > 0:
            folds = make_spatial_folds_buffered(
                df,
                n_splits=args.folds,
                grid_size=args.grid_size,
                seed=seed,
                buffer_cells=args.buffer_cells,
            )
        else:
            folds = make_spatial_folds(df, n_splits=args.folds, grid_size=args.grid_size, seed=seed)
        run_fold_ids = set(range(len(folds)))
        if args.fold_ids.strip():
            run_fold_ids = {int(x) for x in args.fold_ids.split(",") if x.strip()}

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if fold_idx not in run_fold_ids:
                continue
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)

            # P0-1：按折加载特征（与 GNN 同源，第 f 折 CNN 提取）。
            feat_path = gnn_dir / f"node_features_fold_{fold_idx}.pt"
            if not feat_path.exists():
                if not args.cnn_ckpt_dir:
                    raise FileNotFoundError(
                        f"{feat_path.name} 缺失且未给 --cnn_ckpt_dir；请先跑 "
                        f"python -m runner.train.extract_features。"
                    )
                extract_fold_features(
                    df, args.cnn_ckpt_dir, args.processed_dir, feat_path, fold_idx,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            x = torch.load(feat_path, map_location="cpu").numpy()
            if x.shape[0] != len(df):
                raise RuntimeError(
                    f"fold_{fold_idx}: 特征行数 {x.shape[0]} != 元数据行数 {len(df)}（pid 对齐破坏）"
                )

            # Fit scaler on train fold only to avoid leaking validation statistics.
            scaler = StandardScaler().fit(x[train_idx])
            x_scaled_all = scaler.transform(x)
            x_train = x_scaled_all[train_idx]
            x_val = x_scaled_all[val_idx]
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]
            logger.log_data_split(
                {
                    "fold": fold_idx,
                    "seed": seed,
                    "train_size": int(len(df_train)),
                    "val_size": int(len(df_val)),
                    "train_p": int((df_train["set"] == "P").sum()),
                    "train_u": int((df_train["set"] == "U").sum()),
                    "train_rn": int((df_train["set"] == "RN").sum()),
                    "val_p": int((df_val["set"] == "P").sum()),
                    "val_u": int((df_val["set"] == "U").sum()),
                    "val_rn": int((df_val["set"] == "RN").sum()),
                }
            )
            sets_train = np.where(df.iloc[train_idx]["set"].values == "P", 1, 0)
            sets_train[df.iloc[train_idx]["set"].values == "RN"] = 2

            y_train_binary = (df.iloc[train_idx]["set"].values == "P").astype(int)
            y_true_val = (df.iloc[val_idx]["set"].values == "P").astype(int)
            mask_p_val = df.iloc[val_idx]["set"].values == "P"
            mask_rn_val = df.iloc[val_idx]["set"].values == "RN"
            pos_mask_train = y_train_binary == 1

            n_p = max(int(pos_mask_train.sum()), 1)
            pi = float(np.clip(n_p / len(train_idx), 0.01, 0.5))

            baseline_preds = {}
            runtimes = {}

            # uPU
            t0 = time.time()
            try:
                baseline_preds["uPU"] = train_upu_predict(
                    x_train=x_train,
                    train_sets=sets_train,
                    x_val=x_val,
                    pi=pi,
                    epochs=args.upu_epochs,
                    lr=0.01,
                    device=device,
                )
                runtimes["uPU"] = (time.time() - t0) / 60.0
            except Exception as ex:
                negative_rows.append({
                    "exp_id": f"E6_upu_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": "uPU",
                    "fold": fold_idx,
                    "seed": seed,
                    "reason": f"uPU failed: {ex}",
                })

            # Elkan-Noto
            t0 = time.time()
            try:
                baseline_preds["ElkanNoto"] = train_elkan_noto_predict(
                    x_train=x_train,
                    y_train_binary=y_train_binary,
                    x_val=x_val,
                    pos_mask_train=pos_mask_train,
                    seed=seed,
                )
                runtimes["ElkanNoto"] = (time.time() - t0) / 60.0
            except Exception as ex:
                negative_rows.append({
                    "exp_id": f"E6_en_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": "ElkanNoto",
                    "fold": fold_idx,
                    "seed": seed,
                    "reason": f"ElkanNoto failed: {ex}",
                })

            # BaggingPU
            t0 = time.time()
            try:
                baseline_preds["BaggingPU"] = train_bagging_pu_predict(
                    x_train=x_train,
                    pos_mask_train=pos_mask_train,
                    x_val=x_val,
                    seed=seed,
                    bags=12,
                )
                runtimes["BaggingPU"] = (time.time() - t0) / 60.0
            except Exception as ex:
                negative_rows.append({
                    "exp_id": f"E6_bag_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": "BaggingPU",
                    "fold": fold_idx,
                    "seed": seed,
                    "reason": f"BaggingPU failed: {ex}",
                })

            # Label Propagation
            t0 = time.time()
            try:
                lp_pred, lp_converged, lp_n_iter = train_label_prop_predict(
                    x_all=x_scaled_all,
                    train_idx=train_idx,
                    train_sets=sets_train,
                    val_idx=val_idx,
                    max_iter=args.lp_max_iter,
                )
                baseline_preds["LabelPropagation"] = lp_pred
                runtimes["LabelPropagation"] = (time.time() - t0) / 60.0
                if not lp_converged:
                    negative_rows.append({
                        "exp_id": f"E6_lp_fold{fold_idx}_seed{seed}",
                        "stage": "E6",
                        "model": "LabelPropagation",
                        "fold": fold_idx,
                        "seed": seed,
                        "reason": (
                            f"LabelPropagation not converged: "
                            f"n_iter={lp_n_iter}, max_iter={args.lp_max_iter}"
                        ),
                    })
            except Exception as ex:
                negative_rows.append({
                    "exp_id": f"E6_lp_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": "LabelPropagation",
                    "fold": fold_idx,
                    "seed": seed,
                    "reason": f"LabelPropagation failed: {ex}",
                })

            # APPNP
            t0 = time.time()
            try:
                baseline_preds["APPNP"] = train_appnp_predict(
                    x_all=x_scaled_all,
                    edge_index=edge_index,
                    train_idx=train_idx,
                    train_sets=sets_train,
                    val_idx=val_idx,
                    seed=seed,
                    device=device,
                    epochs=args.appnp_epochs,
                )
                runtimes["APPNP"] = (time.time() - t0) / 60.0
            except Exception as ex:
                negative_rows.append({
                    "exp_id": f"E6_appnp_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": "APPNP",
                    "fold": fold_idx,
                    "seed": seed,
                    "reason": f"APPNP failed: {ex}",
                })

            for model_name, pred in baseline_preds.items():
                mp = compute_metric_pack(
                    y_true=y_true_val,
                    y_score=pred,
                    mask_p=mask_p_val,
                    mask_rn=mask_rn_val,
                )
                all_rows.append({
                    "exp_id": f"E6_{model_name}_fold{fold_idx}_seed{seed}",
                    "stage": "E6",
                    "model": model_name,
                    "fold": fold_idx,
                    "seed": seed,
                    "auc": mp.auc,
                    "auc_pu": mp.auc_pu,
                    "auprc": mp.auprc,
                    "brier": mp.brier,
                    "ece": mp.ece,
                    "ece_pu": mp.ece_pu,
                    "sep_prn": mp.sep_prn,
                    "recall_k": mp.recall_k,
                    "rmclp_gain": np.nan,
                    "runtime_min": runtimes.get(model_name, np.nan),
                })
                logger.log_fold_summary(
                    fold=fold_idx,
                    auc=float(mp.auc),
                    auprc=float(mp.auprc),
                    brier=float(mp.brier),
                    sep_prn=float(mp.sep_prn),
                    recall_k=float(mp.recall_k),
                    runtime_min=float(runtimes.get(model_name, np.nan)),
                )

    out_main = out_dir / "results_external.csv"
    out_neg = out_dir / "negative_external.csv"
    pd.DataFrame(all_rows).to_csv(out_main, index=False)
    pd.DataFrame(negative_rows).to_csv(out_neg, index=False)
    logger.log_artifact(path=out_main, artifact_type="results_external_csv", fold=-1)
    logger.log_artifact(path=out_neg, artifact_type="negative_external_csv", fold=-1)
    print(f"Saved baseline method results: {out_main}")
    print(f"Saved baseline method failures: {out_neg}")
    logger.finalize_run(status="success")


if __name__ == "__main__":
    main()
