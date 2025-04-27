"""
Description:
    Loads meteorological station nearest-neighbor distances, extracts sparsely distributed
    stations beyond a specified threshold, overlays them on Chongqing administrative
    boundary with hatched styling, and generates a clear legend highlighting sparse areas.

Inputs:
    - Data/重庆市_市.geojson              : GeoJSON of Chongqing municipal boundary
    - Data/Station_With_Nearest_Distance.csv : CSV of stations with computed nearest neighbor distances

Parameters:
    - DIST_THRESHOLD: float - Distance threshold in meters to define sparse stations (default = 5000)

Outputs:
    - Sparse_Area_Merged_Hatched_Legend_Only.png :
        PNG map highlighting sparse-area stations with hatched red squares.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape, MultiPolygon
from matplotlib.patches import Patch

# --------------------------------------------------
# Configuration: Input file paths and parameters
# --------------------------------------------------
CITY_GEOJSON    = r"Data/重庆市_市.geojson"                       # Chongqing boundary
STATION_CSV     = r"Data/Station_With_Nearest_Distance.csv"      # Station data with distances
DIST_THRESHOLD  = 5000                                            # Sparse threshold (meters)
OUTPUT_FIGURE   = "Sparse_Area_Merged_Hatched_Legend_Only.png"   # Output figure path

# --------------------------------------------------
# Step 1: Load station CSV and validate
# --------------------------------------------------
df = pd.read_csv(STATION_CSV)
if "最近邻距离_m" not in df.columns:
    raise ValueError("CSV missing required field '最近邻距离_m'")

# --------------------------------------------------
# Step 2: Extract sparse-area stations
# --------------------------------------------------
df_sparse = df[df["最近邻距离_m"] > DIST_THRESHOLD].copy()
# Create GeoDataFrame of sparse points
gdf_sparse = gpd.GeoDataFrame(
    df_sparse,
    geometry=[Point(lon, lat) for lon, lat in zip(df_sparse["longitude"], df_sparse["latitude"])],
    crs="EPSG:4326"
)

# --------------------------------------------------
# Step 3: Load and prepare Chongqing boundary
# --------------------------------------------------
with open(CITY_GEOJSON, encoding="utf-8") as f:
    gj = json.load(f)
# Flatten MultiPolygon features
geometries = []
for feat in gj["features"]:
    geom = shape(feat["geometry"])
    if isinstance(geom, MultiPolygon):
        geometries.extend(geom.geoms)
    else:
        geometries.append(geom)
# Choose largest polygon as boundary
boundary = max(geometries, key=lambda g: g.area)

# --------------------------------------------------
# Step 4: Plot boundary and station distributions
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
# Fill Chongqing area in light gray
if boundary.geom_type == "MultiPolygon":
    for poly in boundary.geoms:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='lightgray', alpha=0.6, edgecolor='black', linewidth=0.8)
else:
    x, y = boundary.exterior.xy
    ax.fill(x, y, color='lightgray', alpha=0.6, edgecolor='black', linewidth=0.8)

# Plot all stations as small gray points for context
ax.scatter(
    df["longitude"], df["latitude"],
    c='lightgray', s=5, label=None, alpha=0.6
)

# Plot sparse stations with hatched red squares
gdf_sparse.plot(
    ax=ax,
    marker='s',
    facecolor='none',
    edgecolor='red',
    linewidth=0.8,
    markersize=50,
    hatch='///',
    label=None
)

# --------------------------------------------------
# Step 5: Custom legend and annotations
# --------------------------------------------------
# Create a custom patch for sparse area legend
sparse_patch = Patch(
    facecolor='none',
    edgecolor='red',
    hatch='///',
    label=f"Sparse Region (d > {DIST_THRESHOLD//1000} km)"
)
ax.legend(
    handles=[sparse_patch],
    loc='upper right',
    frameon=True,
    fontsize=10
)

# --------------------------------------------------
# Step 6: Final figure settings
# --------------------------------------------------
ax.set_title(f"Sparse Area (d > {DIST_THRESHOLD//1000} km) on Chongqing Map", fontsize=14)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# --------------------------------------------------
# Step 7: Save and display
# --------------------------------------------------
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.show()
