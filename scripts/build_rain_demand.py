"""
build_rain_demand.py
--------------------
将降雨“需求”（GeoTIFF 或 CSV）映射到每个小栅格（patch）上，生成 per-patch 的需求权重。
支持两种使用方式：
1) 命令行参数（与原版一致）
2) 直接在代码里写 DEFAULTS（未在命令行提供的参数将回退到 DEFAULTS）

输入源（任选其一）：
- GeoTIFF: --rain_tif 指定；多波段会按 --stat 聚合
- CSV    : --rain_csv 指定；两种格式：
    A) 含 pid, shard, cell_id, {demand|value|cv}  -> 直接按键合并
    B) 含 lon, lat, {demand|value|cv}             -> 对每个 patch 中心做最近邻采样

输出：
- demand_per_patch.csv: pid, shard, cell_id, demand_raw, demand（[0,1]）

示例（命令行）：
python build_rain_demand.py \
  --patches_csv data/processed/patches_info.csv \
  --rain_tif data/rain/rx1day_2010_2024.tif \
  --out_csv data/processed/demand_per_patch.csv \
  --stat mean --clip_low 1 --clip_high 99 --norm minmax

示例（直接改 DEFAULTS 后不带任何参数运行）：
python build_rain_demand.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULTS = {
    "patches_csv": r"F:\weather-v2\data\processed\patches_info.csv",
    # 二选一：给 tif 就用 tif；给 csv 就用 csv；若两者都给，优先命令行参数，其次按下面哪个非空就用哪个
    "rain_tif":   r"F:\weather-v2\data\raw\rainfall_cv_chongqing.tif",   # 或 Rx1day.tif
    "rain_csv":   "",  # 例如 r"F:\weather-v2\data\rain\cv_points.csv" 或 含 pid/shard/cell_id 的表
    "out_csv":    r"F:\weather-v2\data\processed\demand_per_patch.csv",
    # 其他超参数
    "stat":       "mean",   # mean / max / min / p95 （仅 tif 有用）
    "clip_low":   1.0,
    "clip_high":  99.0,
    "norm":       "minmax"  # minmax / zscore / log1p
}

# ------------------------- 工具函数 -------------------------

def read_tif_sample_points(tif_path, lons, lats, stat='mean'):
    import rasterio
    src = rasterio.open(tif_path)
    # 按点采样（返回每点的各波段值）
    sampler = src.sample(zip(lons, lats))
    vals = np.array([next(sampler) for _ in range(len(lons))])  # [N, bands] or [N]
    if vals.ndim == 1:
        vals = vals[:, None]
    if stat == 'mean':
        v = np.nanmean(vals, axis=1)
    elif stat == 'max':
        v = np.nanmax(vals, axis=1)
    elif stat == 'min':
        v = np.nanmin(vals, axis=1)
    elif stat == 'p95':
        v = np.nanpercentile(vals, 95, axis=1)
    else:
        v = np.nanmean(vals, axis=1)
    return v

def robust_norm(x, clip_low=1, clip_high=99, method='minmax'):
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, [clip_low, clip_high])
    x = np.clip(x, lo, hi)
    if method == 'minmax':
        mn, mx = np.nanmin(x), np.nanmax(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    elif method == 'zscore':
        mu, sd = np.nanmean(x), np.nanstd(x)
        sd = max(sd, 1e-6)
        z = (x - mu)/sd
        # 将 z 压到 [0,1] 以便与覆盖权重同量纲
        z = 1/(1+np.exp(-z))
        return z
    elif method == 'log1p':
        x = np.log1p(np.maximum(0.0, x - np.nanmin(x)))
        mn, mx = np.nanmin(x), np.nanmax(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn)/(mx - mn)
    else:
        return x

def guess_lon_lat_cols(df):
    lon_col = 'lon' if 'lon' in df.columns else ('x' if 'x' in df.columns else None)
    lat_col = 'lat' if 'lat' in df.columns else ('y' if 'y' in df.columns else None)
    return lon_col, lat_col

def pick_value_column(df):
    # 在 CSV 中猜测代表需求的列名
    for cand in ['demand', 'value', 'cv', 'val', 'rain', 'rx1day']:
        if cand in df.columns:
            return cand
    # 如果只多出一列数值就用它
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError("无法在 rain_csv 里识别需求列，请包含 'demand' 或 'value' 或 'cv' 等列名。")

def sample_from_points_nearest(info, points, val_col):
    """
    rain_csv 模式 B：points 表含 lon/lat 与数值列（val_col），对每个 patch 最近邻采样。
    """
    lon_i, lat_i = guess_lon_lat_cols(info)
    lon_p, lat_p = guess_lon_lat_cols(points)
    if lon_i is None or lat_i is None:
        raise ValueError("patches_csv 需包含 lon/lat 或 x/y 列用以定位。")
    if lon_p is None or lat_p is None:
        raise ValueError("rain_csv 需包含 lon/lat 或 x/y 列（点集）。")

    Xi = info[[lon_i, lat_i]].to_numpy(float)
    Xp = points[[lon_p, lat_p]].to_numpy(float)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(Xp)
    dist, idx = nbrs.kneighbors(Xi, return_distance=True)
    vals = points.iloc[idx[:,0]][val_col].to_numpy()
    return vals

def args_or_defaults():
    """
    混合命令行与 DEFAULTS：
    - 命令行提供的参数优先；
    - 未提供则回退到 DEFAULTS；
    - 若两个源（tif 与 csv）都没给，报错。
    """
    p = argparse.ArgumentParser()
    p.add_argument('--patches_csv', type=str, default=None, help='patches_info.csv')
    p.add_argument('--rain_tif',   type=str, default=None, help='降雨需求 GeoTIFF（单/多波段均可）')
    p.add_argument('--rain_csv',   type=str, default=None, help='降雨需求 CSV（pid/shard/cell_id 或 lon/lat）')
    p.add_argument('--out_csv',    type=str, default=None, help='输出路径')
    p.add_argument('--stat',       type=str, default=None, choices=['mean','max','min','p95'])
    p.add_argument('--clip_low',   type=float, default=None)
    p.add_argument('--clip_high',  type=float, default=None)
    p.add_argument('--norm',       type=str, default=None, choices=['minmax','zscore','log1p'])
    args = p.parse_args()

    # 用 DEFAULTS 填充空值
    cfg = dict(
        patches_csv = args.patches_csv or DEFAULTS["patches_csv"],
        rain_tif    = args.rain_tif    if args.rain_tif is not None else (DEFAULTS["rain_tif"] or ""),
        rain_csv    = args.rain_csv    if args.rain_csv is not None else (DEFAULTS["rain_csv"] or ""),
        out_csv     = args.out_csv     or DEFAULTS["out_csv"],
        stat        = args.stat        or DEFAULTS["stat"],
        clip_low    = args.clip_low    if args.clip_low  is not None else DEFAULTS["clip_low"],
        clip_high   = args.clip_high   if args.clip_high is not None else DEFAULTS["clip_high"],
        norm        = args.norm        or DEFAULTS["norm"],
    )
    # 合法性检查：至少给了一个需求源
    if not cfg["rain_tif"] and not cfg["rain_csv"]:
        raise ValueError("请至少提供一个需求源：--rain_tif 或 --rain_csv（或在 DEFAULTS 中填写）。")
    return cfg

# ------------------------- 主流程 -------------------------

def main():
    cfg = args_or_defaults()

    info = pd.read_csv(cfg["patches_csv"])
    lon_col, lat_col = guess_lon_lat_cols(info)
    if lon_col is None or lat_col is None:
        raise ValueError("patches_csv 需包含 lon/lat 或 x/y 列用以定位。")

    # 1) 从 GeoTIFF 采样
    vals_tif = None
    if cfg["rain_tif"]:
        vals_tif = read_tif_sample_points(
            cfg["rain_tif"],
            info[lon_col].values,
            info[lat_col].values,
            stat=cfg["stat"]
        )

    # 2) 或/并：从 CSV 读取
    vals_csv = None
    if cfg["rain_csv"]:
        r = pd.read_csv(cfg["rain_csv"])
        keys = ['pid','shard','cell_id']
        if all(k in r.columns for k in keys):
            # 模式 A：直接按键合并
            val_col = pick_value_column(r)
            merged = info[keys].merge(r[keys+[val_col]], on=keys, how='left')
            vals_csv = merged[val_col].to_numpy()
        else:
            # 模式 B：lon/lat 最近邻
            val_col = pick_value_column(r)
            vals_csv = sample_from_points_nearest(info, r, val_col)

    # 3) 融合（若两者都给，默认取平均；你也可以改成加权）
    if (vals_tif is not None) and (vals_csv is not None):
        demand_raw = np.nanmean(np.vstack([vals_tif, vals_csv]), axis=0)
    elif vals_tif is not None:
        demand_raw = vals_tif
    else:
        demand_raw = vals_csv

    demand_raw = np.nan_to_num(demand_raw, nan=0.0, posinf=0.0, neginf=0.0)
    demand_norm = robust_norm(demand_raw, clip_low=cfg["clip_low"], clip_high=cfg["clip_high"], method=cfg["norm"])

    out = info[['pid','shard','cell_id']].copy()
    out['demand_raw'] = demand_raw
    out['demand'] = demand_norm
    Path(cfg["out_csv"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out_csv"], index=False)
    print("Saved", cfg["out_csv"], "| mean demand:", float(np.mean(out['demand'])))

if __name__ == '__main__':
    main()
