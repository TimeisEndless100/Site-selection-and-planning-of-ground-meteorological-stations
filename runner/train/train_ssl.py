# -*- coding: utf-8 -*-
"""
Unified SSL pretrain + PU fine-tune script.

Modes:
- simclr: unsupervised contrastive pre-training
- supcon: supervised contrastive pre-training (P/RN/U-aware)
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import argparse
import types
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from src.dataset import PUSpatialDataset
from src.model import GeoConvNeXt
from src.loss import nnpu_loss, bce_with_logits
from src.spatial_split import make_spatial_folds, make_spatial_folds_buffered
from src.contrastive import GeoAugmentation, SimCLR, NTXentLoss
from src.supcon import SupConLoss
from src.core.exp_logger import ExperimentLogger, save_roc_pr_points, save_score_bins
from src.core.metrics import compute_metric_pack
from src.core.rn import assign_sets
from src.core.seed import set_seed


class SupConModel(nn.Module):
    def __init__(self, encoder, head='mlp', dim_in=128, feat_dim=128):
        super().__init__()
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )

    def forward(self, x):
        feat = self.encoder(x)
        return F.normalize(self.head(feat), dim=1)


class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, supcon: bool = False):
        self.base = base_dataset
        self.augment = GeoAugmentation()
        self.supcon = supcon

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        v1 = self.augment(img.clone())
        v2 = self.augment(img.clone())
        if not self.supcon:
            return v1, v2

        row = self.base.df.iloc[idx]
        if row['set'] == 'P':
            label = 1
        elif row['set'] == 'RN':
            label = 0
        else:
            label = -1
        return v1, v2, torch.tensor(label, dtype=torch.long)


def _patch_encoder_forward(base_encoder: GeoConvNeXt) -> None:
    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.head[0](x)
        x = self.head[1](x)
        x = self.head[2](x)
        return x

    base_encoder.forward = types.MethodType(forward_features, base_encoder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="outputs/exp_ssl")
    parser.add_argument("--ssl_mode", default="supcon", choices=["simclr", "supcon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_ids", type=str, default="")
    parser.add_argument("--buffer_cells", type=int, default=0, help="Optional buffered split radius in grid cells")

    parser.add_argument("--epochs_pre", type=int, default=15)
    parser.add_argument("--epochs_ft", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_pre", type=float, default=1e-4)
    parser.add_argument("--lr_ft", type=float, default=1e-4)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--pi_prior", type=float, default=0.15)
    parser.add_argument("--lambda_rn", type=float, default=1.0)

    parser.add_argument("--rn_slope_thresh", type=float, default=45.0)
    parser.add_argument("--rn_water_thresh", type=float, default=0.5)
    parser.add_argument("--log_dir", type=str, default="", help="Enhanced logging dir. Default: <out_dir>/logs")
    parser.add_argument("--disable_enhanced_logging", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    logger = ExperimentLogger(
        enabled=not args.disable_enhanced_logging,
        log_dir=log_dir,
        exp_id=out_dir.name,
        stage="train_ssl",
        model=args.ssl_mode,
        seed=args.seed,
        run_params=vars(args),
        data_paths={"processed_dir": args.processed_dir},
    )

    df = pd.read_csv(Path(args.processed_dir) / "patches_info.csv")
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
            grid_size=10,
            seed=args.seed,
            buffer_cells=args.buffer_cells,
        )
    else:
        folds = make_spatial_folds(df, n_splits=args.folds, grid_size=10, seed=args.seed)
    run_fold_ids = set(range(len(folds)))
    if args.fold_ids.strip():
        run_fold_ids = {int(x) for x in args.fold_ids.split(",") if x.strip()}

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx not in run_fold_ids:
            continue
        fold_start = time.time()
        print(f"\n=== Fold {fold_idx} ({args.ssl_mode}) ===")
        fold_dir = out_dir / f"fold_{fold_idx}"
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
        stats = None

        ds_base = PUSpatialDataset(df_train, args.processed_dir, stats)
        pre_ds = TwoViewDataset(ds_base, supcon=(args.ssl_mode == "supcon"))
        pre_dl = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

        base_encoder = GeoConvNeXt(in_ch=ds_base.input_channels)
        _patch_encoder_forward(base_encoder)

        if args.ssl_mode == "simclr":
            pre_model = SimCLR(encoder=base_encoder, dim_in=128, dim_out=128).to(device)
            criterion_pre = NTXentLoss().to(device)
        else:
            pre_model = SupConModel(base_encoder, dim_in=128, feat_dim=128).to(device)
            criterion_pre = SupConLoss(temperature=args.temp).to(device)

        optimizer_pre = torch.optim.AdamW(pre_model.parameters(), lr=args.lr_pre, weight_decay=1e-4)

        for ep in range(args.epochs_pre):
            ep_start = time.time()
            pre_model.train()
            losses = []
            for batch in tqdm(pre_dl, leave=False, desc=f"Pre Ep {ep}"):
                optimizer_pre.zero_grad()
                if args.ssl_mode == "simclr":
                    v1, v2 = batch
                    v1, v2 = v1.to(device), v2.to(device)
                    _, z1 = pre_model(v1)
                    _, z2 = pre_model(v2)
                    loss = criterion_pre(z1, z2)
                else:
                    v1, v2, labels = batch
                    images = torch.cat([v1, v2], dim=0).to(device)
                    labels = labels.to(device)
                    labels = torch.cat([labels, labels], dim=0)
                    mask_u = labels == -1
                    if mask_u.any():
                        uniq = torch.arange(1000, 1000 + int(mask_u.sum()), device=device)
                        labels[mask_u] = uniq
                    feats = pre_model(images).unsqueeze(1)
                    loss = criterion_pre(feats, labels)
                loss.backward()
                optimizer_pre.step()
                losses.append(loss.item())
            print(f"  Pre Ep {ep}: loss={np.mean(losses):.4f}")
            logger.log_epoch(
                fold=fold_idx,
                epoch=ep,
                phase="ssl_pretrain",
                loss=float(np.mean(losses)),
                lr=float(optimizer_pre.param_groups[0]["lr"]),
                pi=np.nan,
                lambda_rn=float(args.lambda_rn),
                p_score=np.nan,
                rn_score=np.nan,
                time_sec=float(time.time() - ep_start),
            )

        encoder_ckpt = fold_dir / "encoder_ssl.pth"
        torch.save(base_encoder.state_dict(), encoder_ckpt)
        logger.log_artifact(path=encoder_ckpt, artifact_type="encoder_checkpoint", fold=fold_idx)

        # Fine-tune with PURN
        model_ft = GeoConvNeXt(in_ch=ds_base.input_channels).to(device)
        model_ft.load_state_dict(base_encoder.state_dict(), strict=False)
        optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr=args.lr_ft)

        ds_P = PUSpatialDataset(df_train[df_train['set'] == 'P'], args.processed_dir, stats)
        ds_U = PUSpatialDataset(df_train[df_train['set'] == 'U'], args.processed_dir, stats)
        ds_RN = PUSpatialDataset(df_train[df_train['set'] == 'RN'], args.processed_dir, stats)

        dl_P = DataLoader(ds_P, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dl_U = DataLoader(ds_U, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dl_RN = DataLoader(ds_RN, batch_size=args.batch_size, shuffle=True, drop_last=True)
        it_P, it_RN = iter(dl_P), iter(dl_RN)

        for ep in range(args.epochs_ft):
            ep_start = time.time()
            model_ft.train()
            losses = []
            for x_u, _ in tqdm(dl_U, leave=False, desc=f"FT Ep {ep}"):
                try:
                    x_p, _ = next(it_P)
                except StopIteration:
                    it_P = iter(dl_P)
                    x_p, _ = next(it_P)
                try:
                    x_rn, _ = next(it_RN)
                except StopIteration:
                    it_RN = iter(dl_RN)
                    x_rn, _ = next(it_RN)

                x_p, x_u, x_rn = x_p.to(device), x_u.to(device), x_rn.to(device)
                optimizer_ft.zero_grad()
                logits_p = model_ft(x_p)
                logits_u = model_ft(x_u)
                logits_rn = model_ft(x_rn)
                loss_pu = nnpu_loss(logits_p, logits_u, float(args.pi_prior))
                loss_rn = bce_with_logits(logits_rn, torch.zeros_like(logits_rn))
                loss = loss_pu + args.lambda_rn * loss_rn
                loss.backward()
                optimizer_ft.step()
                losses.append(loss.item())
            print(f"  FT Ep {ep}: loss={np.mean(losses):.4f}")
            logger.log_epoch(
                fold=fold_idx,
                epoch=ep,
                phase="ssl_finetune",
                loss=float(np.mean(losses)),
                lr=float(optimizer_ft.param_groups[0]["lr"]),
                pi=float(args.pi_prior),
                lambda_rn=float(args.lambda_rn),
                p_score=np.nan,
                rn_score=np.nan,
                time_sec=float(time.time() - ep_start),
            )

        # Eval
        model_ft.eval()
        df_val_eval = df_val.copy()
        df_val_eval['label'] = (df_val_eval['set'] == 'P').astype(float)
        ds_val = PUSpatialDataset(df_val_eval, args.processed_dir, stats)
        dl_val = DataLoader(ds_val, batch_size=256, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for x, y in dl_val:
                preds.append(torch.sigmoid(model_ft(x.to(device))).cpu().numpy())
                labels.append(y.numpy())
        y_true = np.concatenate(labels)
        y_score = np.concatenate(preds)
        auc = roc_auc_score(y_true, y_score)
        mask_p = (df_val["set"].values == "P")
        mask_rn = (df_val["set"].values == "RN")
        mp = compute_metric_pack(y_true=y_true, y_score=y_score, mask_p=mask_p, mask_rn=mask_rn)
        print(f"  Val AUC: {auc:.4f}")

        model_ckpt = fold_dir / "model_best.pth"
        torch.save(model_ft.state_dict(), model_ckpt)
        logger.log_artifact(path=model_ckpt, artifact_type="model_checkpoint", fold=fold_idx)

        curve_paths = save_roc_pr_points(y_true=y_true, y_score=y_score, out_dir=fold_dir, fold=fold_idx)
        score_bin_path = save_score_bins(set_values=df_val["set"].values, y_score=y_score, out_dir=fold_dir, fold=fold_idx)
        logger.log_artifact(path=curve_paths["roc"], artifact_type="roc_points_csv", fold=fold_idx)
        logger.log_artifact(path=curve_paths["pr"], artifact_type="pr_points_csv", fold=fold_idx)
        logger.log_artifact(path=score_bin_path, artifact_type="score_bins_csv", fold=fold_idx)

        runtime_min = (time.time() - fold_start) / 60.0
        results.append(
            {
                "fold": fold_idx,
                "auc": mp.auc,
                "auprc": mp.auprc,
                "brier": mp.brier,
                "sep_prn": mp.sep_prn,
                "recall_k": mp.recall_k,
                "ssl_mode": args.ssl_mode,
                "seed": args.seed,
                "runtime_min": runtime_min,
            }
        )
        logger.log_fold_summary(
            fold=fold_idx,
            auc=float(mp.auc),
            auprc=float(mp.auprc),
            brier=float(mp.brier),
            sep_prn=float(mp.sep_prn),
            recall_k=float(mp.recall_k),
            runtime_min=float(runtime_min),
        )

    results_path = out_dir / "results_ssl.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    logger.log_artifact(path=results_path, artifact_type="results_csv", fold=-1)
    logger.finalize_run(status="success")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
