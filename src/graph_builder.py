import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def build_grid_graph(df_meta, k=8):
    """
    根据 cell_id 或经纬度构建 8-邻域图
    df_meta: 必须包含 'cell_id' (格式 "latIdx_lonIdx")
    Returns: edge_index [2, Num_Edges]
    """
    print("Building Graph Topology...")

    # 1. 解析坐标索引
    # 假设 cell_id 格式为 "row_col"
    # 如果你的 process.py 生成的 cell_id 是用于 Spatial Split 的粗粒度 grid，
    # 这里我们需要重新解析 pid 或者利用 lat/lon 计算细粒度 grid

    # 更通用的方法：直接利用 lat/lon 找最近邻
    # 这种方法不依赖 grid 对齐，适应性更强
    from sklearn.neighbors import NearestNeighbors

    coords = df_meta[['lat', 'lon']].values

    # k=9 因为包含自己 (nearest is itself)
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    source_nodes = []
    target_nodes = []

    # 2. 构建边
    # indices shape: [N, 9]
    for i in tqdm(range(len(df_meta)), desc="Linking Nodes"):
        # 第0个是自己，跳过
        neighbors = indices[i, 1:]

        # 添加边 (无向图，所以双向添加，或者 PyG 会处理)
        # 这里我们添加 i -> neighbor
        for n_idx in neighbors:
            source_nodes.append(i)
            target_nodes.append(n_idx)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    print(f"Graph built: {len(df_meta)} nodes, {edge_index.shape[1]} edges.")

    return edge_index