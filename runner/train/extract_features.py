import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.dataset import PUSpatialDataset
from src.model import GeoConvNeXt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    # 这里的 model_path 请指向你之前跑出来的效果最好的那个模型 (e.g., outputs/exp_pu_final/fold_0/model_best.pth)
    parser.add_argument("--model_path", required=True, help="Path to the best CNN model checkpoint")
    parser.add_argument("--stats_path", required=True, help="Path to stats.json used for that model")
    parser.add_argument("--out_dir", default="data/gnn_ready")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("Loading metadata...")
    df = pd.read_csv(Path(args.processed_dir) / "patches_info.csv")
    # 确保按 pid 排序，方便后续索引对齐
    df = df.sort_values('pid').reset_index(drop=True)
    df.to_csv(Path(args.out_dir) / "nodes_metadata.csv", index=False)  # 保存对齐后的元数据

    with open(args.stats_path, 'r') as f:
        stats = json.load(f)

    ds = PUSpatialDataset(df, args.processed_dir, stats)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 加载模型并修改为特征提取器
    print(f"Loading CNN from {args.model_path}...")
    model = GeoConvNeXt(in_ch=ds.input_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # --- Hack: 移除分类头，只保留特征输出 ---
    # GeoConvNeXt 的 forward 是: stem -> stage1 -> down -> stage2 -> head
    # head 是: Pool -> Flatten -> Norm -> Linear
    # 我们希望取出 Linear 之前的向量 (128维)
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.m = original_model

        def forward(self, x):
            x = self.m.stem(x)
            x = self.m.stage1(x)
            x = self.m.downsample(x)
            x = self.m.stage2(x)
            # Head 的前三层 (Pool, Flatten, Norm)
            x = self.m.head[0](x)
            x = self.m.head[1](x)
            x = self.m.head[2](x)
            return x  # [B, 128]

    extractor = FeatureExtractor(model).to(device)
    extractor.eval()

    # 3. 提取特征
    features_list = []
    print("Extracting visual features...")
    with torch.no_grad():
        for x, _ in tqdm(dl):
            feat = extractor(x.to(device))
            features_list.append(feat.cpu())

    all_features = torch.cat(features_list, dim=0)
    print(f"Feature shape: {all_features.shape}")

    # 4. 保存
    save_path = Path(args.out_dir) / "node_features.pt"
    torch.save(all_features, save_path)
    print(f"Features saved to {save_path}")


if __name__ == "__main__":
    main()