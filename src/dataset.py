import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class PUSpatialDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processed_dir: Path, stats: dict = None):
        self.df = df.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.stats = stats

        # 预加载第一张图以确定通道数
        if len(self.df) > 0:
            sample_x = self._load_npz(0)
            self.base_channels = sample_x.shape[0]
        else:
            self.base_channels = 0

        # 连续通道索引 (假设前6个是 Slope/DEM)
        # 如果 process.py 输出了更多通道，这里只是用于 z-score 标准化
        self.cont_channels = list(range(min(6, self.base_channels)))

    def _load_npz(self, idx):
        row = self.df.iloc[idx]
        p = self.processed_dir / "patches" / f"shard_{int(row['shard']):03d}" / f"{row['pid']}.npz"
        with np.load(p) as z:
            return z["x"].astype(np.float32)

    def __len__(self):
        return len(self.df)

    @property
    def input_channels(self):
        return self.base_channels

    def __getitem__(self, idx):
        # 1. 加载基础特征
        x = self._load_npz(idx)  # [C, H, W]

        # 2. 统计量标准化 (Z-Score)
        if self.stats:
            for i in self.cont_channels:
                mean = float(self.stats['mean'][i]) if 'mean' in self.stats else 0.0
                std = float(self.stats['std'][i]) if 'std' in self.stats else 1.0
                if std < 1e-6: std = 1.0

                ch = x[i]
                ch = np.nan_to_num(ch, nan=mean, posinf=mean, neginf=mean)
                x[i] = (ch - mean) / std

        x = np.nan_to_num(x, nan=0.0)  # 整体兜底

        # 3. 获取标签 (训练时 label 由 set 决定，或者直接读取 is_P/is_RN)
        # 这里统一返回 label 字段，如果不存在则默认为 0
        label = float(self.df.iloc[idx].get('label', 0.0))

        return torch.from_numpy(x), torch.tensor(label).float()
