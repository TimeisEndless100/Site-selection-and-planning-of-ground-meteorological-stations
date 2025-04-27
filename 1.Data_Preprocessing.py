"""
This script performs geospatial data preprocessing including raster reprojection, clipping, merging multi-band data,
and generating labeled patches for machine learning. It is tailored for remote sensing datasets such as landcover, slope,
and DEM, using Chongqing city as the region of interest.

Dependencies: rasterio, numpy, pandas, shapely, pyproj, fiona, matplotlib

Inputs:
- Landcover raster (.tif)
- Slope raster (.tif)
- DEM raster (.tif)
- Station location CSV file (longitude/latitude)
- Administrative boundary (GeoJSON)

Outputs:
- Unified 3-band raster (.tif)
- Patch images labeled by station coverage
- Patch metadata CSV
- Patch label distribution plot
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from pyproj import Transformer, CRS
import fiona
from shapely.geometry import mapping, shape
import shapely.ops
import matplotlib.pyplot as plt

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def reproject_raster_to_match(infile, out_file, target_crs, ref_transform, ref_width, ref_height):
    """
    Reproject a raster to match a target CRS, transform, and dimensions.

    Args:
        infile (str): Path to input raster.
        out_file (str): Path to output raster.
        target_crs (CRS or str): Target coordinate reference system.
        ref_transform (Affine): Target affine transformation.
        ref_width (int): Target raster width.
        ref_height (int): Target raster height.

    Returns:
        str: Output file path.
    """
    with rasterio.open(infile) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(f"Input raster {infile} has no CRS.")

        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height
        })

        with rasterio.open(out_file, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
    return out_file

def clip_raster_with_geojson(tif_path, geojson_path, out_path):
    """
    Clip a raster file using a GeoJSON administrative boundary.

    Args:
        tif_path (str): Input raster path.
        geojson_path (str): GeoJSON boundary file path.
        out_path (str): Output clipped raster path.

    Returns:
        str: Clipped raster file path.
    """
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs

        with fiona.open(geojson_path, 'r') as shapefile:
            src_crs = shapefile.crs
            shapes = []
            for feature in shapefile:
                geom = shape(feature['geometry'])
                if src_crs and CRS(raster_crs) != CRS(src_crs):
                    transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)
                    geom = shapely.ops.transform(transformer.transform, geom)
                shapes.append(mapping(geom))

        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return out_path

def constrain_landcover_by_dem_slope(multi_band_data, dem_range=(0, 3000), slope_range=(0, 90), nodata=0):
    """
    Apply DEM and slope range constraints to landcover classification.

    Args:
        multi_band_data (ndarray): 3-band stack of landcover, slope, DEM.
        dem_range (tuple): Valid DEM value range.
        slope_range (tuple): Valid slope value range.
        nodata (numeric): Nodata value.

    Returns:
        ndarray: Constrained landcover data.
    """
    landcover, slope, dem = multi_band_data
    slope_min, slope_max = slope_range
    dem_min, dem_max = dem_range
    mask_valid = ((slope >= slope_min) & (slope <= slope_max) & (dem >= dem_min) & (dem <= dem_max))
    constrained_landcover = landcover.copy()
    constrained_landcover[~mask_valid] = nodata
    out_data = multi_band_data.copy()
    out_data[0] = constrained_landcover
    return out_data

def is_patch_valid(patch_data, nodata_value=0, threshold=0.7):
    """
    Check if a patch is valid by measuring valid pixel ratio.

    Args:
        patch_data (ndarray): Patch data.
        nodata_value (numeric): Nodata value.
        threshold (float): Minimum valid pixel ratio.

    Returns:
        bool: True if patch is valid.
    """
    total_pixels = patch_data.shape[1] * patch_data.shape[2]
    valid_pixels = np.count_nonzero(patch_data[0] != nodata_value)
    return (valid_pixels / total_pixels) >= threshold

def unify_and_create_patches(
    landcover_tif, slope_tif, dem_tif, station_csv, geojson_path,
    out_unified_tif="unified_3band.tif", patch_size=64, out_dir="patches--64",
    use_dem_slope_constraint=True, dem_range=(0, 3000), slope_range=(0, 90), nodata_value=0
):
    """
    Merge landcover, slope, and DEM rasters, generate labeled patches.

    Args:
        landcover_tif (str): Landcover raster path.
        slope_tif (str): Slope raster path.
        dem_tif (str): DEM raster path.
        station_csv (str): Station CSV file path.
        geojson_path (str): GeoJSON boundary file path.
        out_unified_tif (str): Output unified raster path.
        patch_size (int): Patch size.
        out_dir (str): Directory to save patches.
        use_dem_slope_constraint (bool): Whether to apply DEM/slope constraints.
        dem_range (tuple): DEM range.
        slope_range (tuple): Slope range.
        nodata_value (numeric): Nodata value.

    Returns:
        None
    """
    # Steps: reproject -> clip -> merge -> patch extraction -> save patches and metadata

    print("[INFO] Reading and aligning rasters...")
    with rasterio.open(landcover_tif) as lc_src:
        target_crs = lc_src.crs
        ref_transform = lc_src.transform
        ref_width = lc_src.width
        ref_height = lc_src.height

    slope_tif_albers = reproject_raster_to_match(slope_tif, "temp_slope_albers.tif", target_crs, ref_transform, ref_width, ref_height)
    dem_tif_albers = reproject_raster_to_match(dem_tif, "temp_dem_albers.tif", target_crs, ref_transform, ref_width, ref_height)

    landcover_tif = clip_raster_with_geojson(landcover_tif, geojson_path, "clipped_landcover.tif")
    slope_tif_albers = clip_raster_with_geojson(slope_tif_albers, geojson_path, "clipped_slope.tif")
    dem_tif_albers = clip_raster_with_geojson(dem_tif_albers, geojson_path, "clipped_dem.tif")

    with rasterio.open(landcover_tif) as lc:
        lc_data = lc.read()
        lc_profile = lc.profile
    with rasterio.open(slope_tif_albers) as sl:
        sl_data = sl.read()
    with rasterio.open(dem_tif_albers) as dm:
        dem_data = dm.read()

    merged_data = np.stack([lc_data[0], sl_data[0], dem_data[0]], axis=0).astype(np.float32)
    if use_dem_slope_constraint:
        merged_data = constrain_landcover_by_dem_slope(merged_data, dem_range=dem_range, slope_range=slope_range, nodata=nodata_value)

    unified_profile = lc_profile.copy()
    unified_profile.update({'count': 3, 'dtype': 'float32'})
    with rasterio.open(out_unified_tif, 'w', **unified_profile) as dst:
        dst.write(merged_data)

    stations_df = pd.read_csv(station_csv)
    stations_df.rename(columns={'编号': 'station_id', '经度': 'longitude', '纬度': 'latitude'}, inplace=True)
    with rasterio.open(out_unified_tif) as src:
        merged_crs = src.crs
        transform = src.transform

    transformer = Transformer.from_crs("EPSG:4326", merged_crs, always_xy=True)
    stations_df["x"], stations_df["y"] = transformer.transform(stations_df["longitude"].values, stations_df["latitude"].values)

    os.makedirs(out_dir, exist_ok=True)
    patches_info = []

    with rasterio.open(out_unified_tif) as src:
        for top_row in range(0, src.height, patch_size):
            for left_col in range(0, src.width, patch_size):
                window = Window(left_col, top_row, min(patch_size, src.width - left_col), min(patch_size, src.height - top_row))
                patch_data = src.read(window=window)
                if np.all(patch_data[0] == nodata_value) or not is_patch_valid(patch_data, nodata_value, threshold=0.7):
                    continue
                label = 0
                for _, st in stations_df.iterrows():
                    x_left, y_top = transform * (left_col, top_row)
                    x_right, y_bottom = transform * (left_col + window.width, top_row + window.height)
                    if x_left <= st["x"] < x_right and y_top >= st["y"] > y_bottom:
                        label = 1
                        break

                patch_filename = f"patch_r{top_row}_c{left_col}_lbl{label}.tif"
                patch_path = os.path.join(out_dir, patch_filename)
                patch_profile = src.profile.copy()
                patch_profile.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, transform)
                })
                with rasterio.open(patch_path, 'w', **patch_profile) as dst:
                    dst.write(patch_data)

                patches_info.append({'row_off': top_row, 'col_off': left_col, 'width': window.width, 'height': window.height, 'label': label, 'filename': patch_filename})

    info_df = pd.DataFrame(patches_info)
    info_df.to_csv(os.path.join(out_dir, "patches_info.csv"), index=False)

    # Draw label distribution bar chart
    label_counts = info_df['label'].value_counts().sort_index()
    labels = ['No Station (0)', 'Station (1)']
    sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, sizes, color=['gray', 'green'])
    plt.title("Patch Label Distribution")
    plt.ylabel("Patch Count")
    for i, v in enumerate(sizes):
        plt.text(i, v + 10, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patch_label_distribution.png"))
    plt.close()

if __name__ == "__main__":
    unify_and_create_patches(
        landcover_tif=r"Data\land cover classification_30m分辨率.tif",
        slope_tif=r"Data\重庆_slope_30m分辨率.tif",
        dem_tif=r"Data\DEM_30m分辨率.tif",
        station_csv=r"Data\Cq_positive_case.csv",
        geojson_path=r"Data\重庆市_市.geojson",
        out_unified_tif="unified_3band.tif",
        patch_size=64,
        out_dir="patches--64-1",
        use_dem_slope_constraint=True,
        dem_range=(0, 3000),
        slope_range=(0, 90),
        nodata_value=0
    )
