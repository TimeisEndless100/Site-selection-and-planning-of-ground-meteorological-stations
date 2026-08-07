import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, BatchNorm


class NodeMLP(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(1)


# 1.  (GAT)
class SpatialGAT(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=1, heads=4):
        super().__init__()
        # Multi-head Attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.3)
        self.bn2 = BatchNorm(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x).squeeze(1)

# 2. 对照组 A: GCN (最基础的谱图卷积)
class SpatialGCN(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=1):
        super().__init__()
        # GCN 比较简单，参数量通常少于 GAT
        self.conv1 = GCNConv(in_channels, hidden_channels * 4) # 放大 hidden 以保持参数量大致公平
        self.bn1 = BatchNorm(hidden_channels * 4)
        self.conv2 = GCNConv(hidden_channels * 4, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x) # GCN 通常用 ReLU
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x).squeeze(1)

# 3. 对照组 B: GraphSAGE (侧重聚合)
class SpatialSAGE(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=1):
        super().__init__()
        # aggregator_type 默认为 'mean'
        self.conv1 = SAGEConv(in_channels, hidden_channels * 4)
        self.bn1 = BatchNorm(hidden_channels * 4)
        self.conv2 = SAGEConv(hidden_channels * 4, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x).squeeze(1)
