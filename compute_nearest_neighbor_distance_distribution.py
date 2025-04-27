"""
Description:
    Reads meteorological station coordinates, computes nearest-neighbor distances using
    a haversine-based BallTree, saves the results to CSV, and visualizes the distance
    distribution as a histogram with key statistical reference lines.

Inputs:
    - Data/Cq_positive_case.csv :
        CSV containing meteorological station records with at least the following columns:
            * 编号       : Station identifier
            * longitude  : Longitude of station (degrees)
            * latitude   : Latitude of station (degrees)

Outputs:
    - Data/Station_With_Nearest_Distance.csv :
        CSV with deduplicated station coordinates and computed nearest-neighbor distance (meters)
    - nearest_neighbor_distribution.png    :
        PNG histogram of nearest-neighbor distances with thresholds and summary lines
"""

import os                    # Operating system utilities for directory creation
import pandas as pd           # Data handling and CSV I/O
import numpy as np            # Numerical operations
from sklearn.neighbors import BallTree  # Efficient nearest-neighbor search on sphere
import matplotlib.pyplot as plt  # Plotting library

# --------------------------------------------------
# Configuration: Create output directory if it does not exist
# --------------------------------------------------
os.makedirs("output", exist_ok=True)

# === Step 1: Read station data and deduplicate coordinates ===
# Load CSV containing station records
df = pd.read_csv("Data/Cq_positive_case.csv")
# Remove duplicate geolocations to avoid zero-distance self matches
df_unique = df.drop_duplicates(subset=["longitude", "latitude"]).copy()
coords = df_unique[["latitude", "longitude"]].to_numpy()

# === Step 2: Compute nearest-neighbor distances using BallTree ===
# Convert coordinates to radians for haversine metric
coords_rad = np.radians(coords)
tree = BallTree(coords_rad, metric='haversine')
# Query each point for nearest neighbor (excluding itself)
dists, _ = tree.query(coords_rad, k=2)
# Convert radian distances to meters (Earth radius ~6371000 m)
nearest_distances_m = dists[:, 1] * 6371000

# === Step 3: Save results to CSV ===
df_unique["最近邻距离_m"] = nearest_distances_m
# Select relevant columns to export
columns_to_save = ["编号", "longitude", "latitude", "最近邻距离_m"]
df_unique[columns_to_save].to_csv(
    "Data/Station_With_Nearest_Distance.csv", index=False
)

# === Step 4: Statistical analysis ===n# Compute mean and median of nearest-neighbor distances
mean_dist = nearest_distances_m.mean()
median_dist = np.median(nearest_distances_m)

# === Step 5: Plot histogram of distances ===
plt.figure(figsize=(10, 6))
# Define bin edges every 1 km
bins = np.arange(0, nearest_distances_m.max() + 1000, 1000)
plt.hist(
    nearest_distances_m,
    bins=bins,
    color='cornflowerblue',
    edgecolor='black'
)
# Add reference lines: density threshold, mean, and median
plt.axvline(x=1000, color='red', linestyle='--', label="Dense Threshold (1 km)")
plt.axvline(x=mean_dist, color='green', linestyle='--', label=f"Mean = {mean_dist:.1f} m")
plt.axvline(x=median_dist, color='orange', linestyle='--', label=f"Median = {median_dist:.1f} m")

# Beautify plot
plt.title("Nearest Neighbor Distance Distribution (Deduplicated Coordinates)")
plt.xlabel("Distance to Nearest Neighbor (meters)")
plt.ylabel("Number of Stations")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# === Step 6: Save and show the figure ===
plt.savefig("nearest_neighbor_distribution.png", dpi=300)
plt.show()
