# -*- coding: utf-8 -*-
"""
数据预处理：Process V2 (适配 PU Spatial Learning)
------------------------------------------------
功能：
1. 多尺度切片 (Multi-scale Cropping): 64x64 (Near) + Context (Mid/Far)
2. 空间对齐 (Spatial Alignment): 自动重投影到 UTM
3. 特征提取 (Feature Extraction): 生成 .npz (图像张量) 和 .csv (元数据)
4. [升级] 统计增强: 计算 slope_max, water_ratio 等物理指标，用于后续 RN 筛选
"""

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from pyproj import Transformer, CRS
from tqdm.auto import tqdm

# 忽略一些且不影响的投影警告
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================
# 配置 (Configuration)
# ==========================
@dataclass
class Config:
    # --- 输入文件路径 ---
    slope_tif: str
    dem_tif: str
    lc_tif: str
    stations_path: Optional[str] = None  # 站点文件 (csv/shp)
    aoi_path: Optional[str] = None  # 感兴趣区域 (可选)

    # --- 输出设置 ---
    out_dir: str = "data/processed"
    shard_size: int = 5000  # 每个文件夹存放多少个文件

    # --- 切片参数 ---
    target_res: float = 30.0  # 目标分辨率 (米)
    patch_size: int = 64  # 核心 patch 大小
    scales: Tuple[int, int, int] = (64, 128, 256)  # 多尺度窗口
    valid_ratio_thresh: float = 0.6  # 有效像素比例阈值 (低于此比例丢弃)

    # --- 标签生成参数 ---
    pos_buffer_m: float = 800.0  # 正例判定半径 (米)

    # --- 土地利用编码 ---
    # CLCD 编码: 1:Crop, 2:Forest, 3:Shrub, 4:Grass, 5:Water, 6:Snow, 7:Barren, 8:Impervious, 9:Wetland
    lc_groups: Dict[str, List[int]] = field(default_factory=lambda: {
        "built": [8],  # 不透水面/建筑
        "crop": [1],  # 农田
        "forest": [2, 3, 4],  # 林/灌/草 (生态用地)
        "water": [5, 9],  # 水体/湿地 (绝对不可建)
        "bare": [7, 6]  # 裸地/冰雪
    })
    lc_nodata: int = 0

    # --- 空间网格 ID 参数 ---
    cell_km: float = 25.0  # 用于生成 cell_id 的网格大小

    # --- CSV 坐标列名 (如果输入是 CSV) ---
    stations_csv_lon: str = "lon"
    stations_csv_lat: str = "lat"


# ==========================
# 辅助函数
# ==========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def best_utm_crs_from_geom(geom_wgs84) -> CRS:
    """根据几何中心自动选择 UTM 投影"""
    minx, miny, maxx, maxy = geom_wgs84.bounds
    cx = 0.5 * (minx + maxx)
    zone = int((cx + 180) // 6) + 1
    # 注意：此处简化默认为北半球，如需全球通用需判断 cy
    epsg = 32600 + zone
    return CRS.from_epsg(epsg)


def read_stations(stations_path: Optional[str], conf: Config, target_crs: CRS) -> Optional[gpd.GeoDataFrame]:
    """读取站点数据并转为目标投影"""
    if not stations_path or not os.path.exists(stations_path):
        return None

    ext = os.path.splitext(stations_path)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            df = pd.read_csv(stations_path)
            # 简单的清洗
            df = df.dropna(subset=[conf.stations_csv_lon, conf.stations_csv_lat])
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[conf.stations_csv_lon], df[conf.stations_csv_lat]),
                crs="EPSG:4326"
            )
        else:
            gdf = gpd.read_file(stations_path)
            if gdf.crs is None:
                raise ValueError("站点矢量文件缺失 CRS 信息")

        return gdf.to_crs(target_crs)
    except Exception as e:
        print(f"[Warn] 读取站点失败: {e}")
        return None


def reproject_to_grid(src_path: str, dst_path: str, dst_crs: CRS,
                      dst_transform: Affine, dst_w: int, dst_h: int,
                      resampling: Resampling, nodata_out, dtype_out=None):
    """重投影并对齐栅格"""
    if os.path.exists(dst_path):
        # 如果已存在且参数一致，可跳过（此处简化，直接覆盖或假设已对齐）
        # print(f"  [Info] {dst_path} exists, skipping reprojection.")
        pass

    ensure_dir(os.path.dirname(dst_path))
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        if dtype_out: profile.update(dtype=dtype_out)
        profile.update({
            'crs': dst_crs, 'transform': dst_transform,
            'width': dst_w, 'height': dst_h,
            'nodata': nodata_out, 'count': 1
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=resampling, dst_nodata=nodata_out
            )
    return dst_path


def calc_cell_id(lon: float, lat: float, lon0: float, lat0: float, cell_km: float) -> str:
    """计算空间网格 ID (用于 Spatial Split)"""
    # 粗略估算：1度纬度 ~= 111km
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))

    lat_idx = int(math.floor((lat - lat0) / (cell_km / km_per_deg_lat)))
    lon_idx = int(math.floor((lon - lon0) / (cell_km / km_per_deg_lon)))
    return f"{lat_idx}_{lon_idx}"


# ==========================
# 主逻辑
# ==========================
def main(conf: Config):
    print(f"=== 开始预处理: Output to {conf.out_dir} ===")
    ensure_dir(conf.out_dir)

    # --- 1. 确定目标坐标系 ---
    # 优先用 AOI，其次用 LC 范围
    ref_geom_wgs = None
    if conf.aoi_path and os.path.exists(conf.aoi_path):
        print("Loading AOI...")
        aoi_gdf = gpd.read_file(conf.aoi_path).to_crs("EPSG:4326")
        ref_geom_wgs = aoi_gdf.geometry.union_all()
    else:
        print("Loading LC bounds as AOI...")
        with rasterio.open(conf.lc_tif) as src:
            left, bottom, right, top = src.bounds
            t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            minx, miny = t.transform(left, bottom)
            maxx, maxy = t.transform(right, top)
            ref_geom_wgs = box(minx, miny, maxx, maxy)

    target_crs = best_utm_crs_from_geom(ref_geom_wgs)
    print(f"Target CRS: {target_crs.to_string()}")

    # 构建对齐网格
    if conf.aoi_path:
        aoi_proj = gpd.GeoSeries([ref_geom_wgs], crs="EPSG:4326").to_crs(target_crs)
        xmin, ymin, xmax, ymax = aoi_proj.total_bounds
    else:
        # 如果用 LC 范围，直接用转换后的坐标
        t_proj = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        minx, miny, maxx, maxy = ref_geom_wgs.bounds
        xmin, ymin = t_proj.transform(minx, miny)
        xmax, ymax = t_proj.transform(maxx, maxy)

    # 计算网格宽高
    width = math.ceil((xmax - xmin) / conf.target_res)
    height = math.ceil((ymax - ymin) / conf.target_res)
    dst_transform = Affine(conf.target_res, 0, xmin, 0, -conf.target_res, ymax)

    # --- 2. 对齐栅格数据 (Alignment) ---
    aligned_dir = os.path.join(conf.out_dir, "aligned")
    ensure_dir(aligned_dir)

    print("Aligning Rasters...")
    path_sl = reproject_to_grid(conf.slope_tif, os.path.join(aligned_dir, "slope.tif"),
                                target_crs, dst_transform, width, height, Resampling.bilinear, np.nan, "float32")
    path_dem = reproject_to_grid(conf.dem_tif, os.path.join(aligned_dir, "dem.tif"),
                                 target_crs, dst_transform, width, height, Resampling.bilinear, np.nan, "float32")
    path_lc = reproject_to_grid(conf.lc_tif, os.path.join(aligned_dir, "lc.tif"),
                                target_crs, dst_transform, width, height, Resampling.nearest, conf.lc_nodata, "uint8")

    # --- 3. 准备站点数据 ---
    stations_gdf = read_stations(conf.stations_path, conf, target_crs)
    stations_union = None
    if stations_gdf is not None:
        print(f"Loaded {len(stations_gdf)} stations.")
        # 创建缓冲区用于判定 is_P
        stations_union = stations_gdf.geometry.buffer(conf.pos_buffer_m).union_all()
    else:
        print("[Warn] No stations loaded. All is_P will be 0.")

    # --- 4. 准备 AOI 过滤 ---
    aoi_union_proj = None
    if conf.aoi_path:
        aoi_union_proj = aoi_proj.union_all()

    # --- 5. 切片循环 (The Loop) ---
    print("Starting Cropping...")
    patches_dir = os.path.join(conf.out_dir, "patches")
    ensure_dir(patches_dir)

    # 预计算 LC 组索引
    lc_keys = list(conf.lc_groups.keys())
    lc_indices = {k: i for i, k in enumerate(lc_keys)}
    water_key_idx = lc_indices.get("water", -1)

    # WGS84 转换器 (用于记录 lon/lat)
    to_wgs84 = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    wgs_bounds = ref_geom_wgs.bounds  # minx, miny, maxx, maxy
    lon0, lat0 = wgs_bounds[0], wgs_bounds[1]

    # 打开所有对齐后的栅格
    with rasterio.open(path_sl) as ds_sl, \
            rasterio.open(path_dem) as ds_dem, \
            rasterio.open(path_lc) as ds_lc:

        # 进度条配置
        margin = (max(conf.scales) - conf.patch_size) // 2
        stride = conf.patch_size
        n_rows = (height - 2 * margin) // stride
        n_cols = (width - 2 * margin) // stride

        info_list = []
        shard_idx = 0
        file_count = 0
        shard_path = os.path.join(patches_dir, f"shard_{shard_idx:03d}")
        ensure_dir(shard_path)

        pbar = tqdm(total=n_rows * n_cols, desc="Processing", unit="patch", mininterval=1.0)

        # 遍历
        for r in range(0, n_rows):
            top = margin + r * stride
            for c in range(0, n_cols):
                left = margin + c * stride
                pbar.update(1)

                # 1. AOI 快速过滤
                # 计算中心点坐标
                cx_idx = left + conf.patch_size // 2
                cy_idx = top + conf.patch_size // 2
                cx, cy = dst_transform * (cx_idx, cy_idx)

                # 如果有 AOI 且中心点不在 AOI 内，跳过
                if aoi_union_proj is not None:
                    if not aoi_union_proj.contains(Point(cx, cy)):
                        continue

                # 2. 读取数据 (Multi-scale)
                # 定义三个窗口：Near (核心), Mid, Far
                # 这里为了简化，我们以 Near 为中心读取
                try:
                    # 读取 Slopes
                    win_near = Window(left, top, conf.patch_size, conf.patch_size)
                    sl_near = ds_sl.read(1, window=win_near)

                    # 检查有效性 (Near 窗口必须有足够多有效值)
                    if np.isnan(sl_near).mean() > (1 - conf.valid_ratio_thresh):
                        continue

                    # 读取 Context (Mid/Far) - 简化逻辑：只取中心对齐
                    # 注意：实际读取时需要根据 scales 计算 offset，这里省略具体坐标计算细节，
                    # 假设 scales=(64, 128, 256)，则 Mid offset=(128-64)/2=32
                    mid_off = (conf.scales[1] - conf.patch_size) // 2
                    far_off = (conf.scales[2] - conf.patch_size) // 2

                    win_mid = Window(left - mid_off, top - mid_off, conf.scales[1], conf.scales[1])
                    win_far = Window(left - far_off, top - far_off, conf.scales[2], conf.scales[2])

                    # 读取并缩放回 64x64
                    def read_and_resize(ds, win, resampling=Resampling.bilinear):
                        arr = ds.read(1, window=win, out_shape=(conf.patch_size, conf.patch_size),
                                      resampling=resampling)
                        return arr

                    sl_mid = read_and_resize(ds_sl, win_mid)
                    sl_far = read_and_resize(ds_sl, win_far)

                    dem_near = ds_dem.read(1, window=win_near)
                    dem_mid = read_and_resize(ds_dem, win_mid)
                    dem_far = read_and_resize(ds_dem, win_far)

                    lc_near = ds_lc.read(1, window=win_near)
                    lc_mid = read_and_resize(ds_lc, win_mid, Resampling.nearest)
                    lc_far = read_and_resize(ds_lc, win_far, Resampling.nearest)

                except Exception:
                    # 边缘处理异常时跳过
                    continue

                # 3. 构建 Tensor (Channel Stacking)
                # Channel 0-2: Slope (Near, Mid, Far)
                # Channel 3-5: DEM (Near, Mid, Far)
                # Channel 6+: LC One-hot (Near, Mid, Far) * Groups

                # LC One-hot 转换
                def make_lc_planes(lc_arr):
                    planes = []
                    for k in lc_keys:
                        mask = np.isin(lc_arr, conf.lc_groups[k])
                        planes.append(mask.astype(np.float32))
                    return np.stack(planes, axis=0)  # [K, H, W]

                lc_planes_near = make_lc_planes(lc_near)
                lc_planes_mid = make_lc_planes(lc_mid)
                lc_planes_far = make_lc_planes(lc_far)

                # 堆叠所有通道
                # 最终形状: [3+3+3*K, 64, 64]
                # 假设 K=5, Total=21 channels
                x = np.concatenate([
                    sl_near[None], sl_mid[None], sl_far[None],
                    dem_near[None], dem_mid[None], dem_far[None],
                    lc_planes_near, lc_planes_mid, lc_planes_far
                ], axis=0).astype(np.float32)

                # 4. [关键] 计算物理统计特征 (用于 RN 筛选)
                # 注意：使用 Near 层的特征来决定当前点是否可建
                slope_val = sl_near  # Near Slope
                dem_val = dem_near  # Near DEM

                # 统计量 (处理 NaN)
                slope_max = float(np.nanmax(slope_val))
                slope_mean = float(np.nanmean(slope_val))
                dem_mean = float(np.nanmean(dem_val))

                # 计算水体占比 (Near 窗口)
                if water_key_idx >= 0:
                    water_ratio = float(lc_planes_near[water_key_idx].mean())
                else:
                    water_ratio = 0.0

                # 5. 生成标签
                # 经纬度
                lon, lat = to_wgs84.transform(cx, cy)
                cell_id = calc_cell_id(lon, lat, lon0, lat0, conf.cell_km)

                # 正例判定 (is_P)
                is_p = 0
                if stations_union is not None:
                    if stations_union.contains(Point(cx, cy)):
                        is_p = 1

                # 6. 保存数据
                pid = f"p_{top:05d}_{left:05d}"
                save_path = os.path.join(shard_path, f"{pid}.npz")
                np.savez_compressed(save_path, x=x)

                # 记录元数据
                info_list.append({
                    "pid": pid,
                    "shard": shard_idx,
                    "cell_id": cell_id,
                    "lon": lon,
                    "lat": lat,
                    "is_P": is_p,
                    # [关键新增] 物理特征，用于 dataset.py 中定义 RN
                    "slope_max": slope_max,
                    "slope_mean": slope_mean,
                    "dem_mean": dem_mean,
                    "water_ratio": water_ratio
                })

                # 分片管理
                file_count += 1
                if file_count >= conf.shard_size:
                    shard_idx += 1
                    file_count = 0
                    shard_path = os.path.join(patches_dir, f"shard_{shard_idx:03d}")
                    ensure_dir(shard_path)

        pbar.close()

    # --- 6. 保存 CSV ---
    print("Saving Metadata...")
    df = pd.DataFrame(info_list)

    # 简单的预处理，防止空值
    df.fillna(0, inplace=True)

    # 为了兼容，可以预先生成一个 'set' 列，但建议在 dataset.py 里动态生成
    # 这里我们只标记 P，其他标记为 U。RN 由 slope_max 等字段在训练时动态决定
    df['set'] = df['is_P'].apply(lambda x: 'P' if x == 1 else 'U')

    csv_path = os.path.join(conf.out_dir, "patches_info.csv")
    df.to_csv(csv_path, index=False)
    print(f"Done. Saved to {csv_path}")
    print(f"Total Patches: {len(df)}")
    print(f"Positives: {df['is_P'].sum()}")


# ==========================
# 运行入口
# ==========================
if __name__ == "__main__":
    # 请根据实际路径修改这里
    cfg = Config(
        # 输入数据
        slope_tif="data/raw/cq_slope_30m.tif",
        dem_tif="data/raw/cq_dem_30m.tif",
        lc_tif="data/raw/CLCD_v01_2024_albert_chongqing.tif",
        stations_path="data/raw/cq_stations.csv",
        aoi_path="data/raw/重庆市_市.geojson",  # 如果有边界文件，建议加上，速度快很多

        # 输出
        out_dir="data/processed",

        # 参数
        target_res=30.0,  # 30米分辨率
        patch_size=64,  # 64x64 像素
        valid_ratio_thresh=0.6,

        # 定义土地利用分类 (CLCD编码)
        lc_groups={
            "cropland": [1],
            "forest": [2],
            "shrub": [3],
            "grassland": [4],
            "water": [5],
            "snow_ice": [6],
            "barren": [7],
            "impervious": [8],
            "wetland": [9]
        }
    )

    main(cfg)