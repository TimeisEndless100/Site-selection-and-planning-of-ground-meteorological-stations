"""
Description:
    Script to visualize predicted probability maps by merging geographic coordinates
    with model prediction probabilities and generating a spatial scatter plot.

Inputs:
    - patches--64/patches_64_with_coords.csv :
        CSV file containing the following columns:
            * filename   : Identifier of each patch image
            * longitude  : Geographic longitude of patch center
            * latitude   : Geographic latitude of patch center

    - patches--64/patches_predictions.csv :
        CSV file containing the following columns:
            * filename   : Identifier matching the coords CSV
            * pred_prob  : Predicted probability value for each patch

Outputs:
    - prediction_probability_map.png :
        High-resolution (300 dpi) PNG image showing spatial distribution of prediction
        probabilities.
"""

import pandas as pd            # Data manipulation library for reading and merging CSV files
import matplotlib.pyplot as plt  # Plotting library for creating figures and scatter plots

# --------------------------------------------------
# Configuration: Ensure proper display of non-ASCII text and minus signs in plots
# --------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font to SimHei for Chinese character support
plt.rcParams['axes.unicode_minus'] = False    # Allow correct rendering of minus signs in axes

# === Step 1: Read input CSV files containing coordinates and predictions ===
# patches_64_with_coords.csv: contains 'filename', 'longitude', 'latitude'
# patches_predictions.csv: contains 'filename', 'pred_prob'
coords_df = pd.read_csv("patches--64/patches_64_with_coords.csv")      # Read coordinate data
preds_df  = pd.read_csv("patches--64/patches_predictions.csv")       # Read prediction probabilities

# === Step 2: Merge coordinate data with prediction probabilities ===
# Use inner join on 'filename' to align coordinate and probability records
merged_df = coords_df.merge(preds_df, on="filename")
# Rename columns to standardized names for plotting
merged_df = merged_df.rename(columns={
    "longitude": "lon",
    "latitude":  "lat",
    "pred_prob": "probability"
})

# === Step 3: Create spatial scatter plot of predicted probabilities ===
plt.figure(figsize=(10, 8))  # Initialize figure with specified dimensions (inches)
sc = plt.scatter(
    merged_df["lon"],            # X-axis: longitude values
    merged_df["lat"],            # Y-axis: latitude values
    c=merged_df["probability"],  # Color-coded by prediction probability
    cmap="viridis",              # Color map for visualizing probability scale
    s=8,                           # Marker size for each point
    alpha=0.8                      # Transparency for overlapping points
)
# Add color bar indicating the mapping from color to probability value
plt.colorbar(sc, label="predicted probability")
# Label axes and add descriptive title (in Chinese and English as needed)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Predicted Probabilities")
# Enhance readability with gridlines
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()  # Adjust layout to prevent clipping

# Save the figure to file for inclusion in manuscripts or presentations
plt.savefig("prediction_probability_map.png", dpi=300)

# Display the plot interactively (useful during exploratory analysis)
plt.show()
