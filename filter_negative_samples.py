"""
Description:
    Filters remote-sensing patch files by their average slope value, retaining only those
    with mean slope greater than a specified threshold (25 degrees) and exports the filtered
    metadata to a new CSV.

Inputs:
    - patches--64/patches_info.csv :
        CSV file listing patch filenames and associated metadata.
    - patches--64/<filename>      :
        Directory containing GeoTIFF patches with three bands: [landcover, slope, elevation].

Outputs:
    - patches--64/patches_info_slope_gt25.csv :
        CSV containing only those records whose slope band average > 25.

Parameters:
    - patch_dir    : str  - Directory containing patch files and metadata CSV
    - threshold    : float - Slope threshold in degrees (default = 25)
"""

import os
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm  # Progress bar for iterations

# ----------------------------
# Configuration
# ----------------------------
patch_dir   = "patches--64"  # Directory with patch files and CSV metadata
csv_path    = os.path.join(patch_dir, "patches_info.csv")
output_csv  = os.path.join(patch_dir, "patches_info_slope_gt25.csv")
threshold   = 25.0             # Average slope threshold (degrees)

# ----------------------------
# Step 1: Load patch metadata
# ----------------------------
print("📥 Loading patch metadata from CSV...")
df = pd.read_csv(csv_path)

# ----------------------------
# Step 2: Iterate and filter by average slope
# ----------------------------
filtered_records = []  # List to store metadata rows that meet threshold
print(f"🔍 Filtering patches with average slope > {threshold}...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering patches"):
    patch_file = os.path.join(patch_dir, row['filename'])
    try:
        # Open multi-band GeoTIFF (bands: 0=landcover, 1=slope, 2=elevation)
        with rasterio.open(patch_file) as src:
            data = src.read()               # Read all bands: shape (3, H, W)
            slope_band = data[1]            # Slope band is index 1
            avg_slope = float(np.nanmean(slope_band))
            # If average slope exceeds threshold, retain this record
            if avg_slope > threshold:
                # Optionally, add slope value to row before appending
                row['avg_slope'] = avg_slope
                filtered_records.append(row)
    except Exception as e:
        # Skip files that cannot be read, log warning
        print(f"⚠️ Skipping file {patch_file}: {e}")

# ----------------------------
# Step 3: Save filtered metadata
# ----------------------------
print(f"📄 Saving {len(filtered_records)} filtered records to CSV...")
filtered_df = pd.DataFrame(filtered_records)
filtered_df.to_csv(output_csv, index=False)

# ----------------------------
# Step 4: Summary
# ----------------------------
print(f"✅ Completed. Original patches: {len(df)}, Filtered: {len(filtered_df)}")
print(f"Filtered metadata saved to: {output_csv}")
