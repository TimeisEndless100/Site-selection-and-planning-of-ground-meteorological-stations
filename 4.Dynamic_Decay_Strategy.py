"""
Description:
    Implements a dynamic probability decay strategy for predicted patch probabilities
    based on distance to existing meteorological stations, and visualizes the adjusted
    spatial probability distribution.

Inputs:
    - patches--64/patches_64_with_coords.csv :
        CSV with columns:
            * filename   : Patch image identifier
            * longitude  : Longitude of patch center (degrees)
            * latitude   : Latitude of patch center (degrees)
    - Data/patches_predictions.csv :
        CSV with columns:
            * filename   : Matching patch identifier
            * pred_prob  : Raw predicted probability from model
    - Data/Cq_positive_case.csv :
        CSV listing existing station cases with columns:
            * latitude   : Station latitude (degrees)
            * longitude  : Station longitude (degrees)

Outputs:
    - adjusted_prediction_probability_map.png :
        PNG image (300 dpi) displaying spatial scatter of adjusted probabilities.
"""

import pandas as pd            # Data manipulation for CSV I/O
import numpy as np             # Numerical operations
from sklearn.neighbors import BallTree  # Fast nearest-neighbor search on sphere
import matplotlib.pyplot as plt  # Plotting

# --------------------------------------------------
# Plot Configuration: Support Chinese labels and minus signs
# --------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese
plt.rcParams['axes.unicode_minus'] = False    # Correct rendering of minus sign

# === Step 1: Load patch coordinates and prediction probabilities ===
# Define file paths
patch_coord_path = "patches--64/patches_64_with_coords.csv"
patch_pred_path  = "Data/patches_predictions.csv"
station_path     = "Data/Cq_positive_case.csv"

# Read CSVs into DataFrames
coords_df = pd.read_csv(patch_coord_path)
preds_df  = pd.read_csv(patch_pred_path)

# Merge on 'filename' to align patch metadata and predictions
merged_df = coords_df.merge(preds_df, on="filename")
# Rename for clarity in later processing
merged_df = merged_df.rename(columns={
    "longitude": "lon",
    "latitude":  "lat",
    "pred_prob":  "probability"
})

# === Step 2: Load existing station coordinates ===
station_df    = pd.read_csv(station_path)
# Extract latitude and longitude as numpy array for BallTree
station_coords = station_df[["latitude", "longitude"]].to_numpy()

# === Step 3: Define dynamic probability decay function ===

def dynamic_decay(d):
    """
    Compute decay adjustment based on distance to nearest station.

    Parameters:
        d (float): Distance from patch to station in meters

    Returns:
        float: Decay value ΔP to subtract from raw probability
    """
    if d <= 0:
        return 0.7
    elif d <= 3000:
        # Linear decay from 0.7 to 0.4 over 0–3 km
        return 0.7 - 0.1 * (d / 1000)
    elif d <= 5000:
        # Linear decay from 0.4 to 0.0 over 3–5 km
        return 0.4 - 0.2 * ((d - 3000) / 1000)
    else:
        return 0.0

# === Step 4: Compute distance from each patch to nearest station ===
# Convert patch and station coords to radians for haversine metric
patch_coords = merged_df[["lat", "lon"]].to_numpy()
tree = BallTree(np.radians(station_coords), metric='haversine')
# Query for nearest neighbor (k=1)
dists_rad, _ = tree.query(np.radians(patch_coords), k=1)
# Convert radian distances to meters (earth radius ~6371000 m)
dists_m = dists_rad[:, 0] * 6371000

# === Step 5: Apply dynamic decay and calculate adjusted probabilities ===
decay_values = np.array([dynamic_decay(d) for d in dists_m])
merged_df["decay"]          = decay_values
# Subtract decay, clip final probabilities between 0 and 1
merged_df["adjusted_prob"]  = np.clip(merged_df["probability"] - merged_df["decay"], 0, 1)

# === Step 6: Visualize adjusted probability spatial distribution ===
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    merged_df["lon"],             # X-axis: longitude
    merged_df["lat"],             # Y-axis: latitude
    c=merged_df["adjusted_prob"], # Color-coded adjusted probability
    cmap='viridis',                 # Color map
    s=8,                            # Marker size
    alpha=0.8                       # Transparency
)
plt.colorbar(sc, label="衰减后概率")  # Color bar label
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("动态概率衰减后的空间分布 / Adjusted Spatial Probability Map")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# Save high-resolution output for publication
plt.savefig("adjusted_prediction_probability_map.png", dpi=300)
plt.show()
