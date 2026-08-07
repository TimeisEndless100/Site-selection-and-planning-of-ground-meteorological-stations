# src/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GeoAugmentation:
    """
    针对地理栅格数据的增强：
    主要使用几何变换（翻转、旋转），不改变物理数值（如坡度、高程）。
    """

    def __call__(self, x):
        # x shape: [C, H, W]

        # 1. 随机水平翻转
        if random.random() > 0.5:
            x = torch.flip(x, dims=[2])

        # 2. 随机垂直翻转
        if random.random() > 0.5:
            x = torch.flip(x, dims=[1])

        # 3. 随机 90度 旋转 (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])

        # 4. (可选) 随机 Cutout / 遮挡
        # 模拟数据缺失，增强鲁棒性
        if random.random() > 0.7:
            h, w = x.shape[1], x.shape[2]
            mask_size = h // 4
            top = random.randint(0, h - mask_size)
            left = random.randint(0, w - mask_size)
            x[:, top:top + mask_size, left:left + mask_size] = 0

        return x


class SimCLR(nn.Module):
    """
    SimCLR 框架包装器
    Encoder: 你的 GeoConvNeXt
    Projector: 投影头 (MLP)
    """

    def __init__(self, encoder, dim_in=128, dim_out=128):
        super().__init__()
        self.encoder = encoder
        # Projector: 映射到隐空间计算 Loss，推理时丢弃
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        # 获取特征 (Representation)
        h = self.encoder(x)  # [B, dim_in]
        # 投影 (Projection)
        z = self.projector(h)
        return h, z


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 拼接 [2N, D]
        z = torch.cat([z_i, z_j], dim=0)

        # 相似度矩阵 [2N, 2N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # 构造标签：对于第 k 个样本，其正例是 k + batch_size (或 k - batch_size)
        labels = torch.cat([
            torch.arange(batch_size, dtype=torch.long, device=z.device) + batch_size,
            torch.arange(batch_size, dtype=torch.long, device=z.device)
        ], dim=0)

        # 屏蔽掉自己与自己的相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss