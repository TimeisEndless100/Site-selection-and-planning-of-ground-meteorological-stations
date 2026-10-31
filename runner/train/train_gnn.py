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
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import subgraph
except ImportError as ex:
    raise ImportError(
        "torch_geometric is required for run_train_gnn.py. "
        "Install dependencies from requirements.txt."
    ) from ex

from src.core.metrics import compute_metric_pack
from src.core.exp_logger import ExperimentLogger, save_roc_pr_points, save_score_bins
from src.core.rn import assign_sets
from src.core.seed import set_seed
from src.gnn_model import NodeMLP, SpatialGAT, SpatialGCN, SpatialSAGE
from src.graph_builder import build_grid_graph
from src.loss import bce_with_logits, nnpu_loss
from src.spatial_split import make_spatial_folds, make_spatial_folds_buffered
from runner.train.extract_features import extract_fold_features, load_pid_sorted_df


def train_epoch(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
    pi: float,
    lambda_rn: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits_all = model(data.x, data.edge_index)
    logits_train = logits_all[train_mask]
    sets_train = data.sets[train_mask]

    mask_p = sets_train == 1
    mask_u = sets_train == 0
    mask_rn = sets_train == 2

    loss_pu = nnpu_loss(logits_train[mask_p], logits_train[mask_u], pi=pi)
    if mask_rn.sum() > 0:
        loss_rn = bce_with_logits(logits_train[mask_rn], torch.zeros_like(logits_train[mask_rn]))
    else:
        loss_rn = torch.tensor(0.0, device=logits_train.device)

    loss = loss_pu + lambda_rn * loss_rn
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_auc_ap(model: torch.nn.Module, data: Data, mask: torch.Tensor) -> tuple[float, float]:
    model.eval()
    probs = torch.sigmoid(model(data.x, data.edge_index))[mask].cpu().numpy()
    y = data.y[mask].cpu().numpy()
    return float(roc_auc_score(y, probs)), float(average_precision_score(y, probs))


@torch.no_grad()
def evaluate_metric_pack(model: torch.nn.Module, data: Data, mask: torch.Tensor | None = None) -> dict[str, float]:
    """评估指标包。mask 非空时只在该节点子集上算指标（P2 修复：transductive 用全图传播 + val_mask）。"""
    model.eval()
    y_score = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()
    y_true = data.y.cpu().numpy()
    sets = data.sets.cpu().numpy()
    if mask is not None:
        m = mask.cpu().numpy()
        y_score, y_true, sets = y_score[m], y_true[m], sets[m]
    mask_p = sets == 1
    mask_rn = sets == 2
    mp = compute_metric_pack(y_true=y_true, y_score=y_score, mask_p=mask_p, mask_rn=mask_rn)
    return {
        "auc": mp.auc,
        "auc_pu": mp.auc_pu,
        "auprc": mp.auprc,
        "brier": mp.brier,
        "ece": mp.ece,
        "ece_pu": mp.ece_pu,
        "sep_prn": mp.sep_prn,
        "recall_k": mp.recall_k,
    }


def build_model(gnn_type: str, in_ch: int, device: torch.device) -> torch.nn.Module:
    if gnn_type == "mlp":
        return NodeMLP(in_channels=in_ch, hidden_channels=64, out_channels=1).to(device)
    if gnn_type == "gat":
        return SpatialGAT(in_channels=in_ch, hidden_channels=64, out_channels=1).to(device)
    if gnn_type == "gcn":
        return SpatialGCN(in_channels=in_ch, hidden_channels=64, out_channels=1).to(device)
    if gnn_type == "sage":
        return SpatialSAGE(in_channels=in_ch, hidden_channels=64, out_channels=1).to(device)
    raise ValueError(f"Unsupported gnn_type: {gnn_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_data_dir", default="data/gnn_ready")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="patches_info.csv 所在目录（自动提取折特征时的元数据源）")
    parser.add_argument("--cnn_ckpt_dir", type=str, default="",
                        help="PURN CNN 主配置训练目录（含 fold_*/model_best.pth + stats.json）；"
                             "折特征文件缺失时用于自动提取（P0-1）")
    parser.add_argument("--out_dir", default="outputs/exp_gnn_compare")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--spatial_grid_size", type=int, default=10)
    parser.add_argument("--buffer_cells", type=int, default=0, help="Optional buffered split radius in grid cells")
    parser.add_argument("--gnn_type", type=str, default="gat", choices=["mlp", "gat", "gcn", "sage"])
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument("--lambda_rn", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_ids", type=str, default="", help="Comma-separated fold ids, e.g. 0,1")
    parser.add_argument("--eval_mode", type=str, default="transductive", choices=["transductive", "inductive"])
    parser.add_argument("--self_loop_only", action="store_true",
                        help="abla_graph 对照：把图替换为纯自环，SAGE 同架构但无节点间传播")
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_dir>/logs")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep previous directory contract: out_dir/<gnn_type>/
    out_dir = Path(args.out_dir) / args.gnn_type
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_dir.name,
        stage="train_gnn",
        model=args.gnn_type,
        seed=args.seed,
        run_params=vars(args),
        data_paths={"gnn_data_dir": args.gnn_data_dir},
    )

    # P0-1（2026-08-12）：不再读全局 node_features.pt。特征在折循环内按折加载
    # （node_features_fold_<f>.pt，由 extract_features 产出或用 --cnn_ckpt_dir 自动提取），
    # 保证 GNN val 节点特征来自排除过它的 CNN，关闭特征级泄露。
    gnn_dir = Path(args.gnn_data_dir)
    meta_path = gnn_dir / "nodes_metadata.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path)  # pid 排序对齐（extract_features 写出的元数据）
    else:
        # 自给：extract_features 未跑时，用 processed 元数据 pid 排序对齐
        df = load_pid_sorted_df(args.processed_dir)
        df.to_csv(meta_path, index=False)

    edge_path = gnn_dir / "edge_index.pt"
    if edge_path.exists():
        edge_index = torch.load(edge_path, map_location="cpu")
    else:
        edge_index = build_grid_graph(df)
        torch.save(edge_index, edge_path)

    if args.self_loop_only:
        # abla_graph「同架构无传播」对照：每个节点只聚合自身，等价于逐节点 MLP
        n = len(df)
        edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
        print(f"  [Info] self_loop_only: 图替换为 {n} 个自环，无节点间传播")

    df = assign_sets(
        df,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy="physics",
        seed=args.seed,
    ).reset_index(drop=True)
    y = torch.tensor(df["is_P"].values, dtype=torch.float32)
    sets = torch.zeros(len(df), dtype=torch.long)
    sets[df["set"] == "P"] = 1
    sets[df["set"] == "RN"] = 2

    if args.buffer_cells > 0:
        folds = make_spatial_folds_buffered(
            df,
            n_splits=args.folds,
            grid_size=args.spatial_grid_size,
            seed=args.seed,
            buffer_cells=args.buffer_cells,
        )
    else:
        folds = make_spatial_folds(df, n_splits=args.folds, grid_size=args.spatial_grid_size, seed=args.seed)
    run_fold_ids = set(range(len(folds)))
    if args.fold_ids.strip():
        run_fold_ids = {int(x) for x in args.fold_ids.split(",") if x.strip()}

    rows: list[dict[str, float | int | str]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx not in run_fold_ids:
            continue

        fold_start = time.time()
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # P0-1：按折加载特征（第 f 折 CNN 提取，排除过该折 val）。缺则用 --cnn_ckpt_dir 自动提取。
        feat_path = gnn_dir / f"node_features_fold_{fold_idx}.pt"
        if not feat_path.exists():
            if not args.cnn_ckpt_dir:
                raise FileNotFoundError(
                    f"{feat_path.name} 缺失且未给 --cnn_ckpt_dir；请先跑 "
                    f"python -m runner.train.extract_features（或经 runner.plan 的 gnn_features 作业）。"
                )
            print(f"  [Info] 缺 {feat_path.name}，用 fold_{fold_idx} CNN checkpoint 提取…")
            extract_fold_features(
                df, args.cnn_ckpt_dir, args.processed_dir, feat_path, fold_idx,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        x = torch.load(feat_path, map_location="cpu")
        if x.size(0) != len(df):
            raise RuntimeError(
                f"fold_{fold_idx}: 特征行数 {x.size(0)} != 元数据行数 {len(df)}（pid 对齐破坏）"
            )
        full_data = Data(x=x, edge_index=edge_index, y=y)
        full_data.sets = sets
        full_data = full_data.to(device)

        train_mask = torch.zeros(len(df), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(len(df), dtype=torch.bool)
        val_mask[val_idx] = True
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        logger.log_data_split(
            {
                "fold": fold_idx,
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

        n_p = int((sets[train_mask] == 1).sum().item())
        n_total = int(train_mask.sum().item())
        pi = max(0.01, n_p / max(n_total, 1))

        model = build_model(args.gnn_type, int(x.shape[1]), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_auc = -np.inf
        best_state = None

        if args.eval_mode == "transductive":
            train_mask_dev = train_mask.to(device)
            val_mask_dev = val_mask.to(device)
            for ep in range(args.epochs):
                ep_start = time.time()
                loss = train_epoch(model, full_data, optimizer, train_mask_dev, pi, lambda_rn=args.lambda_rn)
                if ep % 20 == 0:
                    auc, _ = evaluate_auc_ap(model, full_data, val_mask_dev)
                    if auc > best_auc:
                        best_auc = auc
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                logger.log_epoch(
                    fold=fold_idx,
                    epoch=ep,
                    phase=f"gnn_{args.eval_mode}",
                    loss=float(loss),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    pi=float(pi),
                    lambda_rn=float(args.lambda_rn),
                    p_score=np.nan,
                    rn_score=np.nan,
                    time_sec=float(time.time() - ep_start),
                )

            if best_state is not None:
                model.load_state_dict(best_state)
            ckpt_path = fold_dir / "gnn_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.log_artifact(path=ckpt_path, artifact_type="model_checkpoint", fold=fold_idx)

            # P2 修复：transductive 最终指标 = 全图传播 + val_mask（与模型选择口径一致）。
            # 旧版误用 val 诱导子图评测 → 与 inductive 口径相同，两种协议测的是同一个东西。
            metrics = evaluate_metric_pack(model, full_data, mask=val_mask_dev)
        else:
            train_nodes = torch.tensor(train_idx, dtype=torch.long)
            val_nodes = torch.tensor(val_idx, dtype=torch.long)

            tr_edge, _ = subgraph(train_nodes, edge_index, relabel_nodes=True)
            va_edge, _ = subgraph(val_nodes, edge_index, relabel_nodes=True)

            train_data = Data(x=x[train_nodes], edge_index=tr_edge, y=y[train_nodes])
            train_data.sets = sets[train_nodes]
            train_data = train_data.to(device)

            val_data = Data(x=x[val_nodes], edge_index=va_edge, y=y[val_nodes])
            val_data.sets = sets[val_nodes]
            val_data = val_data.to(device)

            mask_train_all = torch.ones(train_data.x.size(0), dtype=torch.bool, device=device)
            for ep in range(args.epochs):
                ep_start = time.time()
                loss = train_epoch(model, train_data, optimizer, mask_train_all, pi, lambda_rn=args.lambda_rn)
                if ep % 20 == 0:
                    auc, _ = evaluate_auc_ap(
                        model,
                        val_data,
                        torch.ones(val_data.x.size(0), dtype=torch.bool, device=device),
                    )
                    if auc > best_auc:
                        best_auc = auc
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                logger.log_epoch(
                    fold=fold_idx,
                    epoch=ep,
                    phase=f"gnn_{args.eval_mode}",
                    loss=float(loss),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    pi=float(pi),
                    lambda_rn=float(args.lambda_rn),
                    p_score=np.nan,
                    rn_score=np.nan,
                    time_sec=float(time.time() - ep_start),
                )

            if best_state is not None:
                model.load_state_dict(best_state)
            ckpt_path = fold_dir / "gnn_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.log_artifact(path=ckpt_path, artifact_type="model_checkpoint", fold=fold_idx)
            metrics = evaluate_metric_pack(model, val_data)

        if args.eval_mode == "transductive":
            # 曲线/分箱也要用全图传播后的 val 预测，与指标口径一致（P2）
            with torch.no_grad():
                y_score_all = torch.sigmoid(model(full_data.x, full_data.edge_index)).detach().cpu().numpy()
            val_mask_np = val_mask.cpu().numpy()
            y_score_np = y_score_all[val_mask_np]
            y_true_np = full_data.y[val_mask_dev].detach().cpu().numpy()
            sets_np = full_data.sets[val_mask_dev].detach().cpu().numpy()
        else:
            with torch.no_grad():
                y_score_np = torch.sigmoid(model(val_data.x, val_data.edge_index)).detach().cpu().numpy()
            y_true_np = val_data.y.detach().cpu().numpy()
            sets_np = val_data.sets.detach().cpu().numpy()
        set_labels = np.where(sets_np == 1, "P", np.where(sets_np == 2, "RN", "U"))
        curve_paths = save_roc_pr_points(y_true=y_true_np, y_score=y_score_np, out_dir=fold_dir, fold=fold_idx)
        score_bin_path = save_score_bins(set_values=set_labels, y_score=y_score_np, out_dir=fold_dir, fold=fold_idx)
        logger.log_artifact(path=curve_paths["roc"], artifact_type="roc_points_csv", fold=fold_idx)
        logger.log_artifact(path=curve_paths["pr"], artifact_type="pr_points_csv", fold=fold_idx)
        logger.log_artifact(path=score_bin_path, artifact_type="score_bins_csv", fold=fold_idx)

        runtime_min = (time.time() - fold_start) / 60.0
        rows.append(
            {
                "fold": fold_idx,
                "seed": args.seed,
                "eval_mode": args.eval_mode,
                "auc": metrics["auc"],
                "auc_pu": metrics["auc_pu"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "ece_pu": metrics["ece_pu"],
                "sep_prn": metrics["sep_prn"],
                "recall_k": metrics["recall_k"],
                "runtime_min": runtime_min,
            }
        )
        print(
            f"[{args.gnn_type}][{args.eval_mode}] fold={fold_idx} "
            f"auc={metrics['auc']:.4f} auprc={metrics['auprc']:.4f} runtime={runtime_min:.2f}m"
        )
        logger.log_fold_summary(
            fold=fold_idx,
            auc=float(metrics["auc"]),
            auprc=float(metrics["auprc"]),
            brier=float(metrics["brier"]),
            sep_prn=float(metrics["sep_prn"]),
            recall_k=float(metrics["recall_k"]),
            runtime_min=float(runtime_min),
        )

    results_path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    logger.log_artifact(path=results_path, artifact_type="results_csv", fold=-1)
    logger.finalize_run(status="success")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
