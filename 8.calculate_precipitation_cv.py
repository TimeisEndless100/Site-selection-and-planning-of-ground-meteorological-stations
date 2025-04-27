"""
Description:
    Calculates the coefficient of variation (CV) of annual precipitation in the Chongqing
    region over multiple years by aligning GeoTIFF rasters, computing per-pixel
    statistics, visualizing the results, and exporting a CV raster.

Inputs:
    - Data/2019重庆_降雨量.tif        : Annual precipitation for 2019 (GeoTIFF)
    - Data/2020重庆_降雨量.tif        : Annual precipitation for 2020 (GeoTIFF)
    - Data/2021重庆_降雨量.tif        : Annual precipitation for 2021 (GeoTIFF)
    - Data/2022重庆_降雨量2022.tif    : Annual precipitation for 2022 (GeoTIFF)

Outputs:
    - rainfall_cv.tif               : GeoTIFF of per-pixel coefficient of variation
    - Figures displaying standard deviation and CV for exploratory analysis
"""

import os                           # Operating system utilities
import numpy as np                  # Numerical operations
import rasterio                     # Raster I/O
from rasterio.warp import reproject, Resampling  # Raster alignment utilities
import matplotlib.pyplot as plt     # Plotting
from tqdm import tqdm               # Progress bar for loops

# ----------------------------
# Configuration: Define input file paths
# ----------------------------
rainfall_files = [
    r"Data/2019重庆_降雨量.tif",
    r"Data/2020重庆_降雨量.tif",
    r"Data/2021重庆_降雨量.tif",
    r"Data/2022重庆_降雨量2022.tif"
]

# ----------------------------
# Step 1: Load reference raster (2019)
# ----------------------------
print("📥 Loading 2019 reference layer...")
with rasterio.open(rainfall_files[0]) as ref:
    ref_data = ref.read(1).astype(np.float32)
    # Mask non-positive values as NaN
    ref_data[ref_data < 0] = np.nan
    raster_stack = [ref_data]

    # Store spatial metadata for alignment and output
    ref_shape = ref_data.shape
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_profile = ref.profile.copy()

# ----------------------------
# Step 2: Align subsequent rasters to reference
# ----------------------------
print("📐 Aligning subsequent layers to reference...")
for path in tqdm(rainfall_files[1:], desc="Aligning rasters"):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[data < 0] = np.nan
        aligned = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=data,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        raster_stack.append(aligned)

# ----------------------------
# Step 3: Compute per-pixel standard deviation and CV
# ----------------------------
print("📊 Computing standard deviation and CV...")
array_stack = np.stack(raster_stack, axis=0)
std_map = np.nanstd(array_stack, axis=0)
mean_map = np.nanmean(array_stack, axis=0)
# Add small epsilon to avoid division by zero
epsilon = 1e-6
cv_map = std_map / (mean_map + epsilon)

# ----------------------------
# Step 4: Visualize standard deviation and CV
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Standard deviation map
std_img = ax1.imshow(std_map, cmap="viridis")
ax1.set_title("Standard Deviation of Precipitation (2019–2022)")
plt.colorbar(std_img, ax=ax1, fraction=0.046, pad=0.04)

# Coefficient of variation map
cv_img = ax2.imshow(cv_map, cmap="plasma", vmin=0, vmax=1)
ax2.set_title("Coefficient of Variation (CV) of Precipitation")
plt.colorbar(cv_img, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# ----------------------------
# Step 5: Export CV raster to GeoTIFF
# ----------------------------
ref_profile.update(
    dtype=rasterio.float32,
    count=1,
    nodata=np.nan
)
output_cv = "rainfall_cv.tif"
with rasterio.open(output_cv, "w", **ref_profile) as dst:
    dst.write(cv_map.astype(np.float32), 1)

print(f"✅ CV raster successfully saved to: {output_cv}")
