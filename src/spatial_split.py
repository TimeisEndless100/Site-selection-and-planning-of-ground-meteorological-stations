import numpy as np
import pandas as pd


def _prepare_spatial_blocks(df: pd.DataFrame, grid_size: int, seed: int, n_splits: int) -> pd.DataFrame:
    out = df.copy()
    out["lon_bin"] = pd.cut(out["lon"], bins=grid_size, labels=False)
    out["lat_bin"] = pd.cut(out["lat"], bins=grid_size, labels=False)
    # 2026-08-12（P0-1 级联折对齐 + 折划分分层）：
    #  1) 块聚合改用 groupby（按键排序 → 与 df 行序无关）。旧实现 drop_duplicates 的块顺序
    #     依赖 df 行序，CNN（原始序）与 GNN（pid 排序序）会得到不同的 block→fold 分配，
    #     是"级联折不对齐"的根源之一。
    #  2) seed 洗牌保留"不同 seed → 不同折划分"语义；块顺序已固定，故折划分结果与 df 行序无关。
    #  3) 按 P 数量降序稳定排序后 round-robin 分配 → 每折 P 总量均衡（分层），
    #     避免某折 val P 过少 → AUC 未定义 / AUPRC、Brier 跨折不可比。
    blocks = (
        out.groupby(["lon_bin", "lat_bin"], sort=True)
        .agg(p_count=("is_P", "sum"), n_patch=("is_P", "size"))
        .reset_index()
    )
    blocks = blocks.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    blocks = blocks.sort_values("p_count", ascending=False, kind="stable")
    blocks["fold_id"] = np.arange(len(blocks)) % n_splits
    out = out.merge(blocks[["lon_bin", "lat_bin", "fold_id"]], on=["lon_bin", "lat_bin"], how="left")
    return out


def make_spatial_folds(df: pd.DataFrame, n_splits: int = 5, grid_size: int = 10, seed: int = 42):
    """Build spatial folds by assigning each spatial block to one fold."""
    out = _prepare_spatial_blocks(df=df, grid_size=grid_size, seed=seed, n_splits=n_splits)

    folds = []
    print(f"[Spatial Split] {grid_size}x{grid_size} grids, {n_splits} folds.")
    for k in range(n_splits):
        val_indices = out[out["fold_id"] == k].index.to_numpy()
        train_indices = out[out["fold_id"] != k].index.to_numpy()
        n_p_train = out.iloc[train_indices]["is_P"].sum()
        n_p_val = out.iloc[val_indices]["is_P"].sum()
        print(f"  Fold {k}: Train P={n_p_train}, Val P={n_p_val}")
        folds.append((train_indices, val_indices))
    return folds


def make_spatial_folds_buffered(
    df: pd.DataFrame,
    n_splits: int = 5,
    grid_size: int = 10,
    seed: int = 42,
    buffer_cells: int = 1,
):
    """
    Build buffered spatial folds.
    Validation blocks follow the same split logic as `make_spatial_folds`.
    Training excludes blocks within `buffer_cells` neighborhood around validation blocks.
    """
    if buffer_cells <= 0:
        return make_spatial_folds(df=df, n_splits=n_splits, grid_size=grid_size, seed=seed)

    out = _prepare_spatial_blocks(df=df, grid_size=grid_size, seed=seed, n_splits=n_splits)
    folds = []
    print(f"[Spatial Split Buffered] {grid_size}x{grid_size} grids, {n_splits} folds, buffer={buffer_cells} cell(s).")

    for k in range(n_splits):
        val_mask = out["fold_id"] == k
        val_blocks = out.loc[val_mask, ["lon_bin", "lat_bin"]].drop_duplicates().to_numpy()
        if len(val_blocks) == 0:
            folds.append((np.array([], dtype=int), np.array([], dtype=int)))
            continue

        lon_all = out["lon_bin"].to_numpy()
        lat_all = out["lat_bin"].to_numpy()
        near_val = np.zeros(len(out), dtype=bool)

        # Exclude training samples in a Chebyshev neighborhood around validation blocks.
        for lon_b, lat_b in val_blocks:
            near_val |= (np.abs(lon_all - lon_b) <= buffer_cells) & (np.abs(lat_all - lat_b) <= buffer_cells)

        train_mask = (~val_mask.to_numpy()) & (~near_val)
        val_indices = np.where(val_mask.to_numpy())[0]
        train_indices = np.where(train_mask)[0]
        dropped = int((~val_mask.to_numpy()).sum() - train_mask.sum())

        n_p_train = out.iloc[train_indices]["is_P"].sum() if len(train_indices) > 0 else 0
        n_p_val = out.iloc[val_indices]["is_P"].sum() if len(val_indices) > 0 else 0
        print(f"  Fold {k}: Train P={n_p_train}, Val P={n_p_val}, Dropped by buffer={dropped}")
        folds.append((train_indices, val_indices))

    return folds
