"""
Description:
    Identifies and visualizes spatial density classifications (sparse, normal, dense)
    of existing meteorological stations within Chongqing, based on nearest-neighbor distances.

Inputs:
    - Data/Cq_positive_case.csv :
        CSV file containing existing station records with columns:
            * longitude : Station longitude (degrees)
            * latitude  : Station latitude (degrees)

    - Data/重庆市_市.geojson :
        GeoJSON file defining administrative boundary of Chongqing municipality.

Outputs:
    - station_density_classification_map.png :
        High-resolution (300 dpi) PNG image showing station locations classified as:
            * Dense (<1 km to nearest neighbor)
            * Normal (1–5 km)
            * Sparse (>5 km)
"""

import pandas as pd                       # Data manipulation library
import numpy as np                        # Numerical operations
import matplotlib.pyplot as plt           # Plotting library
from sklearn.neighbors import BallTree    # Nearest-neighbor search on spherical coordinates
import json                               # For reading GeoJSON file
from shapely.geometry import shape, MultiPolygon  # Geometry operations

# --------------------------------------------------
# Plot Configuration: Support Chinese characters and minus signs
# --------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese
plt.rcParams['axes.unicode_minus'] = False    # Correct rendering of minus sign

# === Step 1: Read and deduplicate station coordinate data ===
# Load station CSV with 'longitude' and 'latitude' columns
df = pd.read_csv("Data/Cq_positive_case.csv")
# Remove duplicate coordinate entries to avoid zero-distance self matches
df_unique = df.drop_duplicates(subset=["longitude", "latitude"]).copy()
# Extract deduplicated coordinates as numpy array for BallTree
coords = df_unique[["latitude", "longitude"]].to_numpy()

# === Step 2: Compute nearest-neighbor distances using haversine metric ===
# Convert coordinates from degrees to radians
coords_rad = np.radians(coords)
# Build BallTree for efficient neighbor queries on spherical surface
tree = BallTree(coords_rad, metric='haversine')
# Query each point for its two nearest neighbors (self + nearest other)
dists, _ = tree.query(coords_rad, k=2)
# Select second neighbor distance (index 1), convert from radians to meters
nearest_dists = dists[:, 1] * 6371000  # Earth radius ~6371000 m
# Add nearest-neighbor distance as new column
df_unique["nearest_m"] = nearest_dists

# === Step 3: Classify stations by density based on distance thresholds ===
# Initialize all as 'normal'
df_unique["density_type"] = "normal"
# Stations with nearest neighbor < 1 km are 'dense'
df_unique.loc[df_unique["nearest_m"] < 1000, "density_type"] = "dense"
# Stations with nearest neighbor > 5 km are 'sparse'
df_unique.loc[df_unique["nearest_m"] > 5000, "density_type"] = "sparse"

# === Step 4: Load Chongqing administrative boundary from GeoJSON ===
with open("Data/重庆市_市.geojson", encoding="utf-8") as f:
    gj = json.load(f)
# Flatten MultiPolygon into list of Polygon geometries
shapes = []
for feat in gj["features"]:
    geom = shape(feat["geometry"])
    if isinstance(geom, MultiPolygon):
        shapes.extend(geom.geoms)
    else:
        shapes.append(geom)
# Select the largest polygon by area as the main boundary
boundary_poly = max(shapes, key=lambda g: g.area)
boundary_x, boundary_y = boundary_poly.exterior.xy  # Extract coordinates for plotting

# === Step 5: Visualize station locations and density classification ===
plt.figure(figsize=(10, 8))
# Plot administrative boundary outline
plt.plot(boundary_x, boundary_y, color="black", linewidth=1)

# Plot all stations in light gray as 'normal' background points
plt.scatter(
    df_unique["longitude"], df_unique["latitude"],
    c="lightgray", s=10, label="normal area (1–5 km)", alpha=0.6
)
# Plot dense stations in blue (<1 km)
dense_df = df_unique[df_unique["density_type"] == "dense"]
plt.scatter(
    dense_df["longitude"], dense_df["latitude"],
    c="blue", s=12, label="dense area (<1 km)"
)
# Plot sparse stations in red (>5 km)
sparse_df = df_unique[df_unique["density_type"] == "sparse"]
plt.scatter(
    sparse_df["longitude"], sparse_df["latitude"],
    c="red", s=12, label="sparse area (>5 km)"
)

# Configure legend, labels, and gridlines
plt.legend(title="density classification", loc="upper left", frameon=True)
plt.title("Station Density Classification")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")  # Maintain aspect ratio for geographic accuracy

# Save high-resolution figure for publication
plt.savefig("station_density_classification_map.png", dpi=300)
plt.show()
