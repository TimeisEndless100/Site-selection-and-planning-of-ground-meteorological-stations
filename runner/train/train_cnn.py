# -*- coding: utf-8 -*-
"""
训练脚本：PU Learning / PN Learning / Taylor Robust PU (Enhanced Logging & Viz)
功能：
1. 动态定义可靠负例 (RN)
2. 空间交叉验证 (Spatial CV)
3. 训练模式支持：pu, pn, taylor
4. [新增] 自动绘制 Loss 曲线、ROC/PR 曲线、得分分布图
5. [新增] 计算 Brier Score, Recall@K, P-RN 分离度等高级指标
"""
import sys
from pathlib import Path
import matplotlib
# 设置无头模式，防止在服务器上报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 将项目根目录加入 path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, auc
)

from src.dataset import PUSpatialDataset
from src.model import GeoConvNeXt
from src.loss import nnpu_loss, bce_with_logits, taylor_pu_loss
from src.spatial_split import make_spatial_folds, make_spatial_folds_buffered
from src.core.exp_logger import ExperimentLogger, save_roc_pr_points, save_score_bins
from src.core.rn import assign_sets
from src.core.seed import set_seed


# ================= 配置 =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="outputs/exp_v1_enhanced")
    parser.add_argument("--mode", type=str, default="pu", choices=["pu", "pn", "taylor"])

    # RN 筛选
    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument("--rn_strategy", type=str, default="physics", choices=["physics", "none", "random", "noisy"])
    parser.add_argument("--rn_noise_rate", type=float, default=0.0, help="When rn_strategy=noisy, flip RN labels at this rate")
    parser.add_argument("--lambda_rn", type=float, default=1.0, help="Weight for RN penalty term")

    # 训练超参
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--spatial_grid_size", type=int, default=10)
    parser.add_argument("--buffer_cells", type=int, default=0, help="Optional buffered split radius in grid cells")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (Windows spawn); 0 = single-process. 实测 num_workers=2 + persistent_workers 在本机 (RTX3060, 内存吃紧) 最优")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--nnpu_epochs", type=int, default=15)
    parser.add_argument("--pi_prior", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_ids", type=str, default="", help="Comma-separated fold ids to run, e.g. 0,1")

    parser.add_argument("--disable_warmup", action="store_true")
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_dir>/logs")
    parser.add_argument("--disable_enhanced_logging", action="store_true")

    return parser.parse_args()


# ================= 绘图与评估工具 =================

def plot_training_history(history_df, save_path):
    """绘制 Loss 和 Score 变化曲线"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss (左轴)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history_df['epoch'], history_df['loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 绘制 Scores (右轴) - 仅当存在 score 列时
    if 'p_score' in history_df.columns:
        ax2 = ax1.twinx()
        color_p = 'tab:green'
        color_rn = 'tab:blue'
        ax2.set_ylabel('Predicted Probability', color='black')
        ax2.plot(history_df['epoch'], history_df['p_score'], color=color_p, linestyle='--', label='Avg P Score')
        ax2.plot(history_df['epoch'], history_df['rn_score'], color=color_rn, linestyle='--', label='Avg RN Score')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 1.05)

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

    plt.title('Training Dynamics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_eval_curves(y_true, y_score, save_path):
    """绘制 ROC 和 PR 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color='green', lw=2, label=f'AP = {pr_auc:.4f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_score_distribution(df_val, y_score, save_path):
    """绘制 P, U, RN 的预测得分分布直方图"""
    plt.figure(figsize=(10, 6))

    # 将预测分放回 DataFrame
    df_plot = df_val.copy()
    df_plot['score'] = y_score

    # 绘制分布
    sns.kdeplot(data=df_plot[df_plot['set']=='P'], x='score', fill=True, label='Positive (P)', color='green', alpha=0.3)
    sns.kdeplot(data=df_plot[df_plot['set']=='U'], x='score', fill=True, label='Unlabeled (U)', color='gray', alpha=0.3)
    sns.kdeplot(data=df_plot[df_plot['set']=='RN'], x='score', fill=True, label='Reliable Neg (RN)', color='red', alpha=0.3)

    plt.xlim(0, 1)
    plt.title('Predicted Score Distribution')
    plt.xlabel('Buildability Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 辅助计算 =================

def compute_stats(df, processed_dir, limit=3000):
    """计算用于标准化的均值方差"""
    if len(df) == 0: return {}
    ds = PUSpatialDataset(df.sample(min(len(df), limit)), processed_dir)
    if len(ds) == 0: return {}
    sample_x, _ = ds[0]
    n_channels = min(6, sample_x.shape[0])
    sums = np.zeros(n_channels)
    sq_sums = np.zeros(n_channels)
    cnt = 0
    for i in range(len(ds)):
        x, _ = ds[i]
        x = x.numpy()[:n_channels]
        sums += x.sum(axis=(1, 2))
        sq_sums += (x ** 2).sum(axis=(1, 2))
        cnt += 64 * 64
    if cnt == 0: return {}
    mean = sums / cnt
    var = np.maximum(sq_sums / cnt - mean ** 2, 0)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}

def estimate_pi(model, df_P, processed_dir, stats, device, num_workers=0):
    model.eval()
    if len(df_P) == 0: return 0.1
    ds = PUSpatialDataset(df_P, processed_dir, stats=stats)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers)
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    if not probs: return 0.1
    return np.mean(np.concatenate(probs))


def _val_auc(model, dl_val, device) -> float:
    """验证集 ROC-AUC（P=1，其余=0），用于逐 epoch 最优模型选择（P5）。

    验证集无 P（或全 P）时 AUC 未定义，返回 NaN（调用方跳过该 epoch 的选优）。
    num_workers=0：验证集仅 ~4.4k patch，且需逐 epoch 复用，避免每次迭代重开 worker
    （Windows spawn 慢）或 persistent worker 长期占内存（本机内存吃紧）。
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dl_val:
            preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            labels.append(y.numpy())
    if not preds:
        return float("nan")
    y_true = np.concatenate(labels)
    y_score = np.concatenate(preds)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ================= 主流程 =================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_epochs = 0 if args.disable_warmup else args.warm_epochs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_dir.name,
        stage="train_cnn",
        model=args.mode,
        seed=args.seed,
        run_params=vars(args),
        data_paths={"processed_dir": args.processed_dir},
    )

    print(f"Loading info from {args.processed_dir}...")
    df = pd.read_csv(Path(args.processed_dir) / "patches_info.csv")

    # 1. 动态定义 RN
    print("Defining RNs...")
    df = assign_sets(
        df,
        rn_slope_thresh=args.rn_slope_thresh,
        rn_water_thresh=args.rn_water_thresh,
        rn_strategy=args.rn_strategy,
        rn_noise_rate=args.rn_noise_rate,
        seed=args.seed,
    )

    print(f"Dataset Split: P={sum(df['set'] == 'P')}, RN={sum(df['set'] == 'RN')}, U={sum(df['set'] == 'U')}")

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
        run_fold_ids = {int(x) for x in args.fold_ids.split(",") if x.strip() != ""}
    results = []

    # --- 循环每一折 ---
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx not in run_fold_ids:
            continue
        fold_start = time.time()
        print(f"\n=== Fold {fold_idx} (Mode: {args.mode}) ===")
        fold_dir = Path(args.out_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
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

        # 统计量
        stats = compute_stats(df_train, args.processed_dir)
        with open(fold_dir / "stats.json", "w") as f: json.dump(stats, f)

        # 训练 Dataset 标签设置 (P=1, others=0)
        df_train['label'] = (df_train['set'] == 'P').astype(float)

        # 初始化模型
        temp_ds = PUSpatialDataset(df_train.iloc[:10], args.processed_dir, stats)
        model = GeoConvNeXt(
            in_ch=temp_ds.input_channels,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # 验证集（P=1，其余=0）：训练前构建一次，供逐 epoch 选优复用（P5）。
        df_val_eval = df_val.copy()
        df_val_eval['label'] = (df_val_eval['set'] == 'P').astype(float)
        ds_val = PUSpatialDataset(df_val_eval, args.processed_dir, stats)
        dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=0)
        best_auc = -1.0
        best_state = None
        best_epoch = -1

        # 日志记录器
        training_history = []

        # ====================================================
        # 模式 A: Naive PN
        # ====================================================
        if args.mode == 'pn':
            print("  [Mode: Naive PN] Treating all U/RN as Negatives...")
            df_train_pn = df_train.copy()
            df_train_pn.loc[df_train_pn['set'] != 'P', 'label'] = 0.0
            df_train_pn.loc[df_train_pn['set'] == 'P', 'label'] = 1.0

            ds_pn = PUSpatialDataset(df_train_pn, args.processed_dir, stats)
            dl_pn = DataLoader(ds_pn, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)

            for ep in range(warmup_epochs + args.nnpu_epochs):
                ep_start = time.time()
                model.train()
                losses = []
                for x, y in tqdm(dl_pn, leave=False, desc=f"PN Ep {ep}"):
                    optimizer.zero_grad()
                    loss = bce_with_logits(model(x.to(device)), y.to(device))
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                avg_loss = np.mean(losses)
                val_auc = _val_auc(model, dl_val, device)
                if not np.isnan(val_auc) and val_auc > best_auc:
                    best_auc, best_epoch = val_auc, ep
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  Ep {ep}: PN Loss = {avg_loss:.4f} | valAUC = {val_auc:.4f}")
                training_history.append({'epoch': ep, 'loss': avg_loss, 'p_score': np.nan, 'rn_score': np.nan})
                logger.log_epoch(
                    fold=fold_idx,
                    epoch=ep,
                    phase="pn",
                    loss=float(avg_loss),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    pi=np.nan,
                    lambda_rn=np.nan,
                    p_score=np.nan,
                    rn_score=np.nan,
                    time_sec=float(time.time() - ep_start),
                )

        # ====================================================
        # 模式 B & C: PU Learning (Standard / Taylor)
        # ====================================================
        else:
            # --- Phase 1: Warm-up ---
            df_warm = df_train[df_train['set'].isin(['P', 'RN'])].copy()
            if warmup_epochs > 0 and len(df_warm[df_warm['set'] == 'RN']) > 0:
                print("  Warm-up with P vs RN...")
                df_warm.loc[df_warm['set']=='P', 'label'] = 1.0
                df_warm.loc[df_warm['set']=='RN', 'label'] = 0.0
                ds_warm = PUSpatialDataset(df_warm, args.processed_dir, stats)
                dl_warm = DataLoader(ds_warm, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

                for ep in range(warmup_epochs):
                    ep_start = time.time()
                    model.train()
                    losses = []
                    for x, y in tqdm(dl_warm, leave=False, desc=f"Warm Ep {ep}"):
                        optimizer.zero_grad()
                        loss = bce_with_logits(model(x.to(device)), y.to(device))
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                    # 记录 Warm-up 阶段
                    training_history.append({'epoch': ep, 'loss': np.mean(losses), 'p_score': np.nan, 'rn_score': np.nan})
                    logger.log_epoch(
                        fold=fold_idx,
                        epoch=ep,
                        phase="warmup",
                        loss=float(np.mean(losses)),
                        lr=float(optimizer.param_groups[0]["lr"]),
                        pi=np.nan,
                        lambda_rn=float(args.lambda_rn),
                        p_score=np.nan,
                        rn_score=np.nan,
                        time_sec=float(time.time() - ep_start),
                    )
            elif args.disable_warmup:
                print("  [Info] Warm-up disabled for this run.")

            # --- Phase 2: Pi Estimation ---
            if args.pi_prior:
                pi = args.pi_prior
            else:
                c_hat = estimate_pi(model, df_train[df_train['set'] == 'P'], args.processed_dir, stats, device, num_workers=args.num_workers)
                p_obs = len(df_train[df_train['set'] == 'P']) / len(df_train)
                pi = np.clip(p_obs / (c_hat + 1e-6), 0.01, 0.5)
            print(f"  Estimated Prior pi = {pi:.4f}")
            with open(fold_dir / "pi.txt", "w") as f: f.write(str(pi))

            # --- Phase 3: Main Training ---
            print(f"  Starting Training (Mode: {args.mode})...")
            ds_P = PUSpatialDataset(df_train[df_train['set'] == 'P'], args.processed_dir, stats)
            ds_U = PUSpatialDataset(df_train[df_train['set'] == 'U'], args.processed_dir, stats)
            ds_RN = PUSpatialDataset(df_train[df_train['set'] == 'RN'], args.processed_dir, stats)

            if len(ds_P) == 0 or len(ds_U) == 0:
                raise RuntimeError(
                    f"Fold {fold_idx} has empty P/U set for PU training: "
                    f"P={len(ds_P)}, U={len(ds_U)}"
                )

            use_rn_penalty = len(ds_RN) > 0 and args.lambda_rn > 0
            if len(ds_RN) == 0:
                print("  [Info] No RN samples in this fold. RN penalty is disabled for this run.")

            dl_P = DataLoader(ds_P, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, persistent_workers=args.num_workers > 0)
            dl_U = DataLoader(ds_U, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, persistent_workers=args.num_workers > 0)
            dl_RN = None
            if use_rn_penalty:
                dl_RN = DataLoader(ds_RN, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, persistent_workers=args.num_workers > 0)
            iter_P = iter(dl_P)
            iter_RN = iter(dl_RN) if dl_RN is not None else None

            for ep in range(warmup_epochs, warmup_epochs + args.nnpu_epochs):
                ep_start = time.time()
                model.train()
                losses = []
                losses_pu = []
                losses_rn = []
                for x_u, _ in tqdm(dl_U, leave=False, desc=f"Main Ep {ep}"):
                    try: x_p, _ = next(iter_P)
                    except: iter_P = iter(dl_P); x_p, _ = next(iter_P)
                    x_p, x_u = x_p.to(device), x_u.to(device)
                    optimizer.zero_grad()

                    logits_p = model(x_p)
                    logits_u = model(x_u)

                    if args.mode == 'taylor':
                        loss_main = taylor_pu_loss(logits_p, logits_u)
                    else:
                        loss_main = nnpu_loss(logits_p, logits_u, pi)

                    if use_rn_penalty:
                        try:
                            x_rn, _ = next(iter_RN)
                        except StopIteration:
                            iter_RN = iter(dl_RN)
                            x_rn, _ = next(iter_RN)
                        x_rn = x_rn.to(device)
                        logits_rn = model(x_rn)
                        loss_rn = bce_with_logits(logits_rn, torch.zeros_like(logits_rn))
                    else:
                        loss_rn = torch.tensor(0.0, device=device)

                    loss = loss_main + args.lambda_rn * loss_rn
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    # 分量累计（loss_pu/loss_rn 拆分，支撑 λ 量级论证）
                    losses_pu.append(loss_main.item())
                    losses_rn.append(loss_rn.item())

                # Check Scores
                model.eval()
                with torch.no_grad():
                    sample_p, _ = ds_P[0]
                    p_score = float(torch.sigmoid(model(sample_p.unsqueeze(0).to(device))).item())
                    if use_rn_penalty:
                        sample_rn, _ = ds_RN[0]
                        rn_score = float(torch.sigmoid(model(sample_rn.unsqueeze(0).to(device))).item())
                    else:
                        rn_score = np.nan

                avg_loss = np.mean(losses)
                avg_loss_pu = float(np.mean(losses_pu)) if losses_pu else 0.0
                avg_loss_rn = float(np.mean(losses_rn)) if losses_rn else 0.0
                val_auc = _val_auc(model, dl_val, device)
                if not np.isnan(val_auc) and val_auc > best_auc:
                    best_auc, best_epoch = val_auc, ep
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                rn_txt = f"{rn_score:.2f}" if not np.isnan(rn_score) else "nan"
                print(f"  Ep {ep}: Loss={avg_loss:.4f} | PU={avg_loss_pu:.4f}, RN={avg_loss_rn:.4f} | P={p_score:.2f}, RN={rn_txt} | valAUC={val_auc:.4f}")

                training_history.append({
                    'epoch': ep, 'loss': avg_loss,
                    'loss_pu': avg_loss_pu, 'loss_rn': avg_loss_rn,
                    'p_score': p_score, 'rn_score': rn_score
                })
                logger.log_epoch(
                    fold=fold_idx,
                    epoch=ep,
                    phase="main",
                    loss=float(avg_loss),
                    loss_pu=avg_loss_pu,
                    loss_rn=avg_loss_rn,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    pi=float(pi),
                    lambda_rn=float(args.lambda_rn),
                    p_score=float(p_score),
                    rn_score=float(rn_score),
                    time_sec=float(time.time() - ep_start),
                )

        # --- Save Training History ---
        training_log_path = fold_dir / "training_log.csv"
        loss_curve_path = fold_dir / "loss_curve.png"
        history_df = pd.DataFrame(training_history, columns=["epoch", "loss", "loss_pu", "loss_rn", "p_score", "rn_score"])
        history_df.to_csv(training_log_path, index=False)
        if len(history_df) > 0:
            plot_training_history(history_df, loss_curve_path)
        logger.log_artifact(path=training_log_path, artifact_type="training_log", fold=fold_idx)
        if loss_curve_path.exists():
            logger.log_artifact(path=loss_curve_path, artifact_type="loss_curve_png", fold=fold_idx)

        # --- P5: 恢复验证集最优模型（逐 epoch valAUC 选优），再进最终评测 ---
        # model_best.pth 语义由"末轮模型"改为"验证集最优模型"（与 train_gnn.py 同口径）
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  [P5] Loaded best model from epoch {best_epoch} (valAUC={best_auc:.4f})")
        else:
            print("  [P5] 无有效验证集 AUC，回退为末轮模型")

        # --- Final Evaluation (Comprehensive) ---
        model.eval()

        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in dl_val:
                val_preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
                val_labels.append(y.numpy())

        y_true = np.concatenate(val_labels)
        y_score = np.concatenate(val_preds)

        # 1. 基础指标
        auc_val = roc_auc_score(y_true, y_score)
        ap_val = average_precision_score(y_true, y_score)

        # 2. 校准度 Brier Score
        brier = brier_score_loss(y_true, y_score)

        # 3. P-RN Separation
        # 从原始 df_val 中获取 P 和 RN 的 mask (y_score 顺序与 df_val 一致)
        mask_p = (df_val['set'] == 'P').values
        mask_rn = (df_val['set'] == 'RN').values
        score_p_mean = y_score[mask_p].mean() if mask_p.any() else np.nan
        if mask_rn.any():
            score_rn_mean = y_score[mask_rn].mean()
            separation = score_p_mean - score_rn_mean
        else:
            score_rn_mean = np.nan
            separation = np.nan

        # 4. Recall@K (选址命中率)
        # 假设我们选 Top 10% 的 P 数量
        k = int(mask_p.sum())
        if k > 0:
            top_k_indices = np.argsort(y_score)[::-1][:k]
            recall_at_k = y_true[top_k_indices].sum() / k
        else:
            recall_at_k = 0.0

        print(f"  Validation Stats:")
        print(f"    ROC-AUC: {auc_val:.4f} | AP: {ap_val:.4f}")
        print(f"    Brier Score: {brier:.4f} (Lower is better)")
        print(f"    P-RN Sep: {separation:.4f} (P={score_p_mean:.2f}, RN={score_rn_mean:.2f})")
        print(f"    Recall@K (K={k}): {recall_at_k:.4f}")

        # 绘图
        roc_pr_curve_png = fold_dir / "roc_pr_curve.png"
        score_dist_png = fold_dir / "score_dist.png"
        plot_eval_curves(y_true, y_score, roc_pr_curve_png)
        plot_score_distribution(df_val, y_score, score_dist_png)
        curve_paths = save_roc_pr_points(y_true=y_true, y_score=y_score, out_dir=fold_dir, fold=fold_idx)
        score_bin_path = save_score_bins(set_values=df_val["set"].values, y_score=y_score, out_dir=fold_dir, fold=fold_idx)
        logger.log_artifact(path=roc_pr_curve_png, artifact_type="roc_pr_curve_png", fold=fold_idx)
        logger.log_artifact(path=score_dist_png, artifact_type="score_distribution_png", fold=fold_idx)
        logger.log_artifact(path=curve_paths["roc"], artifact_type="roc_points_csv", fold=fold_idx)
        logger.log_artifact(path=curve_paths["pr"], artifact_type="pr_points_csv", fold=fold_idx)
        logger.log_artifact(path=score_bin_path, artifact_type="score_bins_csv", fold=fold_idx)

        # 保存模型
        model_path = fold_dir / "model_best.pth"
        torch.save(model.state_dict(), model_path)
        logger.log_artifact(path=model_path, artifact_type="model_checkpoint", fold=fold_idx)

        # 汇总结果
        final_pi = 0 if args.mode == 'pn' else (args.pi_prior if args.pi_prior else pi)
        runtime_min = (time.time() - fold_start) / 60.0
        results.append({
            "fold": fold_idx,
            "auc": auc_val,
            "ap": ap_val,
            "brier": brier,
            "separation": separation,
            "recall_k": recall_at_k,
            "pi": final_pi,
            "runtime_min": runtime_min,
        })
        logger.log_fold_summary(
            fold=fold_idx,
            auc=float(auc_val),
            auprc=float(ap_val),
            brier=float(brier),
            sep_prn=float(separation),
            recall_k=float(recall_at_k),
            runtime_min=float(runtime_min),
        )

    # 保存最终摘要
    results_path = Path(args.out_dir) / "results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logger.log_artifact(path=results_path, artifact_type="results_csv", fold=-1)
    logger.finalize_run(status="success")
    print("Training Finished! Check 'outputs/' for logs and plots.")


if __name__ == "__main__":
    main()
