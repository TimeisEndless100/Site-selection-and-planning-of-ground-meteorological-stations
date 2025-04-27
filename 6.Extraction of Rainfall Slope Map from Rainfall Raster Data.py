"""
Description:
    Reads annual rainfall raster data (GeoTIFF), computes spatial gradient (slope)
    using Sobel operator, visualizes the gradient map with geographic coordinates,
    and exports both a high-resolution PNG and a new GeoTIFF of gradient values.

Inputs:
    - Data/2022重庆_降雨量2022.tif :
        Single-band GeoTIFF of annual rainfall (e.g., mm/year), with nodata values.

Outputs:
    - Rainfall_Gradient_Sobel_Coordinates.png :
        PNG image (300 dpi) showing spatial gradient magnitude with proper
        geographic axes and colorbar.
    - Rainfall_Gradient_Sobel.tif :
        GeoTIFF of computed gradient magnitude, preserving original spatial profile.
"""

import numpy as np               # Array operations
import rasterio                  # Reading and writing raster (GeoTIFF) data
from matplotlib import pyplot as plt  # Plotting
from scipy import ndimage        # Image processing (Sobel filter)
import rasterio.plot             # Utilities for geospatial plotting

# --------------------------------------------------
# Plot Configuration: Enable Chinese labels and correct minus signs
# --------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese text
plt.rcParams['axes.unicode_minus'] = False    # Render minus sign correctly

# === Step 1: Read annual rainfall raster and handle nodata ===
# Open the GeoTIFF file containing annual rainfall data
rain_src = rasterio.open("Data/2022重庆_降雨量2022.tif")
# Read band 1 into numpy array
rain_data = rain_src.read(1)
# Replace nodata values with NaN for proper gradient computation
rain_data = np.where(rain_data == rain_src.nodata, np.nan, rain_data)

# === Step 2: Compute gradient magnitude using Sobel operator ===
# Apply Sobel filter along X (longitude) axis
sobel_x = ndimage.sobel(rain_data, axis=1, mode='constant', cval=np.nan)
# Apply Sobel filter along Y (latitude) axis
sobel_y = ndimage.sobel(rain_data, axis=0, mode='constant', cval=np.nan)
# Combine gradients to get magnitude (approximate slope)
rain_gradient = np.hypot(sobel_x, sobel_y)

# === Step 3: Visualize gradient with geographic coordinates ===
fig, ax = plt.subplots(figsize=(10, 6))
# Obtain geographic extent (left, right, bottom, top) for proper axis
extent = rasterio.plot.plotting_extent(rain_src)
# Display gradient raster with colormap
im = ax.imshow(rain_gradient, cmap="inferno", extent=extent)
# Add colorbar with label in Chinese
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Gradient Magnitude")
# Set title and axis labels
ax.set_title("Rainfall Gradient Map")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# Save high-resolution PNG for manuscript
gradient_coord_vis_path = "Rainfall_Gradient_Sobel_Coordinates.png"
plt.savefig(gradient_coord_vis_path, dpi=300)
plt.close()

# === Step 4: Write gradient magnitude to new GeoTIFF ===
# Prepare profile based on source raster
gradient_profile = rain_src.profile.copy()
# Update profile: float32 data type, single band, set nodata to NaN
gradient_profile.update({
    "dtype": "float32",
    "count": 1,
    "nodata": np.nan
})
# Write out gradient array as GeoTIFF
gradient_raster_path = "Rainfall_Gradient_Sobel.tif"
with rasterio.open(gradient_raster_path, "w", **gradient_profile) as dst:
    dst.write(rain_gradient.astype(np.float32), 1)

# Output path for downstream use
print(f"Gradient GeoTIFF saved to: {gradient_raster_path}")
