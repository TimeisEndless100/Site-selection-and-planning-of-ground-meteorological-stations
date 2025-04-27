"""
Description:
    Extracts centroids of connected high-gradient regions from a rainfall gradient GeoTIFF
    based on a specified threshold, exports centroid coordinates to CSV, and visualizes
    the points on the gradient map using marker '×'.

Inputs:
    - Data/Rainfall_Gradient_Sobel.tif :
        GeoTIFF file of rainfall gradient magnitude computed via Sobel operator.

Parameters:
    - threshold_gradient_25 (float):
        Gradient magnitude threshold for region extraction (e.g., 25 units).

Outputs:
    - High_Rainfall_Gradient_Points_gt25.csv :
        CSV file listing longitude and latitude of each extracted centroid.
    - High_Rainfall_Gradient_Points_Map_gt25_markerX.png :
        PNG image (300 dpi) visualizing gradient map with '×' markers at centroids.
"""

import matplotlib                              # Core plotting settings
import numpy as np                             # Array operations
import pandas as pd                            # Tabular data handling
import rasterio                                # Raster I/O
import geopandas as gpd                        # Vector geospatial data handling
import matplotlib.pyplot as plt                # Plotting functions
from shapely.geometry import Point             # Constructing geometric points
from skimage.measure import label, regionprops # Connected-component analysis
from rasterio.plot import plotting_extent      # Get geographic extent for plotting

# --------------------------------------------------
# Plot Configuration: Chinese font and minus sign support
# --------------------------------------------------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # SimHei font for Chinese
matplotlib.rcParams['axes.unicode_minus'] = False    # Correct minus sign

# === Step 1: Define input raster path and read gradient data ===
gradient_tif_path = r"Data/Rainfall_Gradient_Sobel.tif"  # Input gradient GeoTIFF
with rasterio.open(gradient_tif_path) as src:
    rain_gradient = src.read(1)           # Read gradient band
    rain_transform = src.transform        # Affine transform for coordinate conversion
    extent = plotting_extent(src)         # Geographic extent for plotting

# === Step 2: Generate binary mask of high-gradient regions ===
threshold_gradient_25 = 25                # Threshold for gradient magnitude
# Create boolean mask where gradient > threshold
high_gradient_mask_25 = rain_gradient > threshold_gradient_25

# Label connected regions in mask (background=0)
label_high_25 = label(high_gradient_mask_25)
# Compute region properties including centroid
regions_high_25 = regionprops(label_high_25)

# === Step 3: Extract centroid coordinates in geographic space ===
high_gradient_coords_25 = []              # List to store centroid lon/lat
for region in regions_high_25:
    cy, cx = region.centroid              # Centroid in array coordinates (row, col)
    # Convert array indices to geographic coordinates (lon, lat)
    lon, lat = rain_transform * (cx, cy)
    high_gradient_coords_25.append((lon, lat))

# === Step 4: Save centroids as CSV using GeoDataFrame ===
gdf_grad_25 = gpd.GeoDataFrame(
    geometry=[Point(xy) for xy in high_gradient_coords_25],
    crs="EPSG:4326"
)
# Extract coordinate columns for export
gdf_grad_25["longitude"] = gdf_grad_25.geometry.x
gdf_grad_25["latitude"]  = gdf_grad_25.geometry.y
csv_grad_25 = "High_Rainfall_Gradient_Points_gt25.csv"
gdf_grad_25[["longitude", "latitude"]].to_csv(csv_grad_25, index=False)

# === Step 5: Visualize gradient map and centroids ===
fig, ax = plt.subplots(figsize=(10, 6))
# Display gradient raster with 'inferno' colormap
img = ax.imshow(rain_gradient, cmap="inferno", extent=extent)
# Scatter centroids with cyan '×' markers
x_lon, x_lat = zip(*high_gradient_coords_25)
ax.scatter(
    x_lon, x_lat,
    c="cyan", marker='x', s=8,
    label=f"梯度 > {threshold_gradient_25}"
)
# Add colorbar for gradient magnitude
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Gradient Magnitude")
# Set titles and labels

ax.set_title(f"Rainfall_Gradient_Points(> {threshold_gradient_25})")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.legend(loc="best")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# Save figure for publication
png_grad_25 = "High_Rainfall_Gradient_Points_Map_gt25_markerX.png"
plt.savefig(png_grad_25, dpi=300)
plt.close()

# === Step 6: Print output file paths ===
print(f"✅ 已完成！\nCSV: {csv_grad_25}\nPNG: {png_grad_25}")
