# -*- coding: utf-8 -*-
"""按折提取 PURN-CNN 特征（P0-1 级联折对齐，2026-08-12）。

背景：旧实现用单折 checkpoint 对全域提一份全局 ``node_features.pt``，导致 GNN 的
val 节点特征来自一个训练过这些节点的 CNN —— 特征级泄露，GNN 桶数字被高估。

新协议（本模块为唯一生产者，按折产出，全部按 pid 排序对齐）：

- 训练/评测（``runner.train.train_gnn``、``runner.train.baselines``）消费
  ``node_features_fold_<f>.pt``：由第 f 折 CNN checkpoint（训练时排除该折 val）提取，
  val 节点特征来自"没见过它"的模型 → 泄露关闭。
- 推理/应用（``runner.train.infer_gnn_all``、``analysis.fold4``）继续消费全局
  ``node_features.pt``（``--global_from_fold`` 指定用哪折 CNN 出图；整域推理无留出
  声明，合法）。

用法（薄 CLI，多折循环；也可 import ``extract_fold_features`` 复用于 train_gnn）：
    python -m runner.train.extract_features \\
        --ckpt_dir outputs/.../cmp_main_cnn/purn_s42 \\
        --processed_dir data/processed --out_dir data/gnn_ready \\
        --folds 5 --global_from_fold 0
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import json
import argparse
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.dataset import PUSpatialDataset
from src.model import GeoConvNeXt


class FeatureExtractor(nn.Module):
    """GeoConvNeXt 去掉分类头，只保留 128 维特征输出（head 前三层 Pool→Flatten→Norm）。"""

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.m = original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.m.stem(x)
        x = self.m.stage1(x)
        x = self.m.downsample(x)
        x = self.m.stage2(x)
        x = self.m.head[0](x)  # Pool
        x = self.m.head[1](x)  # Flatten
        x = self.m.head[2](x)  # Norm
        return x  # [B, 128]


def load_pid_sorted_df(processed_dir: str | Path) -> pd.DataFrame:
    """全量 patch 元数据，按 pid 排序（特征/图/标签行序的唯一锚）。"""
    df = pd.read_csv(Path(processed_dir) / "patches_info.csv")
    return df.sort_values("pid").reset_index(drop=True)


def extract_fold_features(
    df: pd.DataFrame,
    cnn_ckpt_dir: str | Path,
    processed_dir: str | Path,
    out_path: str | Path,
    fold_idx: int,
    device: str = "cuda",
    batch_size: int = 256,
    num_workers: int = 4,
) -> torch.Tensor:
    """用第 ``fold_idx`` 折的 CNN checkpoint 对全量 patch 提 128 维特征并缓存。

    关键保证：``cnn_ckpt_dir/fold_<f>/model_best.pth`` 训练时**排除**了该折 val，
    故特征对 val 节点无标签泄露。stats 取同一折的 ``fold_<f>/stats.json``。
    行序由传入 ``df``（pid 排序）保证，与特征/图/标签行序唯一对齐。
    """
    fold_dir = Path(cnn_ckpt_dir) / f"fold_{fold_idx}"
    ckpt_path = fold_dir / "model_best.pth"
    stats_path = fold_dir / "stats.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint 缺失: {ckpt_path}（需先跑 cmp_main_cnn 的 PURN 格）"
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    ds = PUSpatialDataset(df, processed_dir, stats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = GeoConvNeXt(in_ch=ds.input_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    extractor = FeatureExtractor(model).to(device)
    extractor.eval()

    feats = []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc=f"Extract fold {fold_idx}"):
            feats.append(extractor(x.to(device)).cpu())
    all_features = torch.cat(feats, dim=0)
    if all_features.size(0) != len(df):
        raise RuntimeError(
            f"fold_{fold_idx}: 特征行数 {all_features.size(0)} != 元数据行数 "
            f"{len(df)}（对齐破坏，pid 排序失效）"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_features, out_path)
    return all_features


def main():
    parser = argparse.ArgumentParser(description="按折提取 PURN-CNN 特征（P0-1 级联折对齐）")
    parser.add_argument(
        "--ckpt_dir", required=True,
        help="PURN CNN 主配置训练目录（含 fold_*/model_best.pth 与 fold_*/stats.json）",
    )
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="data/gnn_ready")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_ids", type=str, default="", help="逗号分隔 fold ids；空 = 全部")
    parser.add_argument(
        "--global_from_fold", type=int, default=None,
        help="同时写全局 node_features.pt（推理/诊断用），指定用哪折 CNN 出图",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pid_sorted_df(args.processed_dir)
    df.to_csv(out_dir / "nodes_metadata.csv", index=False)  # pid 排序对齐元数据（幂等）

    run_fold_ids = (
        {int(x) for x in args.fold_ids.split(",") if x.strip()}
        if args.fold_ids.strip()
        else set(range(args.folds))
    )
    for f in sorted(run_fold_ids):
        out_path = out_dir / f"node_features_fold_{f}.pt"
        if out_path.exists():
            print(f"[skip] {out_path.name} 已存在（重跑需先手动删除）")
            continue
        extract_fold_features(
            df, args.ckpt_dir, args.processed_dir, out_path, f,
            device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print(f"-> {out_path}")

    if args.global_from_fold is not None:
        gpath = out_dir / "node_features.pt"
        if not gpath.exists():
            extract_fold_features(
                df, args.ckpt_dir, args.processed_dir, gpath, args.global_from_fold,
                device=device, batch_size=args.batch_size, num_workers=args.num_workers,
            )
            print(f"全局特征（推理用, fold_{args.global_from_fold} CNN）-> {gpath}")


if __name__ == "__main__":
    main()
