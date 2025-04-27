"""
Description:
    Implements a greedy maximum coverage algorithm (MRAM-NetOpt) for meteorological station
    deployment by integrating candidate and existing station points, optimizing coverage,
    determining replaceable stations, and visualizing replacement mappings with custom legends.

Inputs:
    - candidate_csv                 : CSV of high-probability patches in sparse regions
    - existing_csv                  : CSV of current meteorological station locations
    - demand_csv                    : CSV of demand points extracted from rainfall gradient centroids
    - cv_raster_path                : GeoTIFF of rainfall coefficient of variation (CV)
    - boundary_geojson_path         : GeoJSON of Chongqing administrative boundary

Outputs:
    - selected_optimized_stations.csv : Selected station locations after optimization
    - replacement_mapping.csv         : Mapping of selected stations to replaced existing stations
    - replacement_visualizationstations_custom_legend.png : Visualization of replacement relationships

Parameters:
    - R_COVER                       : Service radius for coverage (meters)
    - MAX_STATIONS                  : Maximum number of stations to select
    - REPLACEMENT_RADIUS            : Distance threshold for replacement eligibility (meters)
    - REPLACEABLE_THRESHOLD         : Distance threshold defining replaceable existing stations (meters)
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import shape, MultiPolygon
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sympy.abc import alpha  # symbolic parameter placeholder
import matplotlib

# === Parameter Configuration ===
R_COVER = 10000             # Service radius for coverage (m)
MAX_STATIONS = 100          # Maximum number of stations to select
REPLACEMENT_RADIUS = 10000  # Max distance to consider replacing an existing station (m)
REPLACEABLE_THRESHOLD = 1000# Distance threshold for identifying replaceable existing stations (m)

# === Input and Output Paths ===
candidate_csv = r"Data/High_Prob_Patches_in_Sparse_Region.csv"
existing_csv  = r"Data/Cq_positive_case.csv"
demand_csv    = r"Data/Combined_Gradient_gt25_gt50_AllPoints.csv"
cv_raster_path = r"Data/rainfall_cv_chongqing.tif"
boundary_geojson_path = r"Data/重庆市_市.geojson"

output_selected_csv   = "selected_optimized_stations.csv"
output_replacement_csv= "replacement_mapping.csv"
output_figure_dodgerblue = "replacement_visualizationstations_custom_legend.png"

# === Utility Functions ===
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points on Earth (meters).

    Parameters:
        lat1, lon1: float - coordinates of point 1 in degrees
        lat2, lon2: float - coordinates of point 2 in degrees
    Returns:
        float: Distance in meters
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def visualize_replacement_custom_legend(boundary_geojson_path, df_replace,
                                        replaceable_coords, selected_coords,
                                        title, color, save_path,title_fontsize):
    """
    Visualize replacement relationships between selected and replaceable stations.
    Draws Chongqing boundary, station markers, and dashed lines connecting replacements.

    Parameters:
        boundary_geojson_path: str - path to Chongqing GeoJSON boundary
        df_replace: DataFrame - mapping of selected and replaced station coords
        replaceable_coords: np.ndarray - coords of replaceable existing stations
        selected_coords: np.ndarray - coords of selected new stations
        title: str - plot title
        color: str - color for replaceable station markers
        save_path: str - output PNG file path
    """
    # Load administrative boundary and extract largest polygon
    with open(boundary_geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    geometries = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"])
        if isinstance(geom, MultiPolygon):
            geometries.extend(geom.geoms)
        else:
            geometries.append(geom)
    polygon = max(geometries, key=lambda g: g.area)
    x, y = polygon.exterior.xy

    # Plot boundary and station points
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y, color="black", linewidth=1)
    ax.scatter(replaceable_coords[:,0], replaceable_coords[:,1],
               c=color, s=12, marker='x', alpha=0.6, label="可替代站点")
    ax.scatter(selected_coords[:,0], selected_coords[:,1],
               c="crimson", s=14, marker='x', label="优化站点")
    # Draw replacement lines
    for _, row in df_replace.iterrows():
        ax.plot([row["selected_lon"], row["replaced_lon"]],
                [row["selected_lat"], row["replaced_lat"]],
                linestyle='--', color='gray', linewidth=0.8)
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, label='Chongqing Boundary'),
        Line2D([0], [0], marker='x', color=color, linestyle='None', markersize=7, label='Replaceable Station'),
        Line2D([0], [0], marker='x', color='crimson', linestyle='None', markersize=7, label='Selected Station'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Main Execution ===
if __name__ == '__main__':
    # Load candidate, existing, and demand points
    df_candidates = pd.read_csv(candidate_csv)
    df_existing   = pd.read_csv(existing_csv)
    df_demand     = pd.read_csv(demand_csv)

    # Extract coordinates arrays
    coords_candidates = df_candidates[['longitude','latitude']].values
    coords_existing   = df_existing[['longitude','latitude']].values
    coords_demand     = df_demand[['longitude','latitude']].values

    # Identify replaceable existing stations by nearest-neighbor distance
    tree_exist = BallTree(np.radians(coords_existing), metric='haversine')
    dists_exist, _ = tree_exist.query(np.radians(coords_existing), k=2)
    nearest_distances = dists_exist[:,1] * 6371000
    replaceable_mask = nearest_distances < REPLACEABLE_THRESHOLD
    coords_replaceable = coords_existing[replaceable_mask]

    # Filter new candidates outside existing station radius
    dists_cand, _ = tree_exist.query(np.radians(coords_candidates), k=1)
    mask_new = (dists_cand[:,0]*6371000) > 500
    coords_new_candidates = coords_candidates[mask_new]
    coords_mixed = np.vstack([coords_new_candidates, coords_replaceable])

    # Build coverage sets N_i and M_j
    N_i, M_j = {}, {}
    for i, (lon_i, lat_i) in enumerate(coords_demand):
        covered = [
        j for j, (lon_j, lat_j) in enumerate(coords_mixed)
        # Calculate distance using haversine(lat, lon, lat, lon)
        if haversine(lat_i, lon_i, lat_j, lon_j) <= R_COVER
    ]
        if covered:
            N_i[i] = covered
            for j in covered:
                M_j.setdefault(j, []).append(i)

    # Load CV raster for demand weighting
    with rasterio.open(cv_raster_path) as src:
        cv_data = src.read(1)
        cv_transform = src.transform
        cv_nodata = src.nodata
    weights = []
    for lon, lat in coords_demand:
        row, col = ~cv_transform * (lon, lat)
        row, col = int(row), int(col)
        if 0<=row<cv_data.shape[0] and 0<=col<cv_data.shape[1]:
            v = cv_data[row,col]
            weights.append(0.0 if v==cv_nodata or np.isnan(v) else float(v))
        else:
            weights.append(0.0)
    weights = np.array(weights)
    weights = weights/weights.sum() if weights.sum()>0 else np.ones_like(weights)

    # Greedy selection
    selected, covered = set(), set()
    while len(selected) < MAX_STATIONS:
        best_j, best_gain = None, 0
        for j, covers in M_j.items():
            if j in selected: continue
            gain = sum(weights[i] for i in covers if i not in covered)
            if gain > best_gain:
                best_gain = gain; best_j = j
        if best_j is None or best_gain==0: break
        selected.add(best_j)
        covered.update(M_j[best_j])

    # Save selected stations
    selected_coords = coords_mixed[list(selected)]
    pd.DataFrame(selected_coords, columns=['longitude','latitude']).to_csv(output_selected_csv, index=False)

    # Map replacements for visualization
    tree_replace = BallTree(np.radians(coords_replaceable), metric='haversine')
    replacements=[]
    for lon,lat in selected_coords:
        dist, idx = tree_replace.query(np.radians([[lon,lat]]), k=1)
        if dist[0][0]*6371000 <= REPLACEMENT_RADIUS:
            repl = coords_replaceable[idx[0][0]]
            replacements.append({'selected_lon':lon,'selected_lat':lat,
                                 'replaced_lon':repl[0],'replaced_lat':repl[1]})
    df_replace = pd.DataFrame(replacements)
    df_replace.to_csv(output_replacement_csv, index=False)

    # Generate replacement visualization
    visualize_replacement_custom_legend(
        boundary_geojson_path,
        df_replace,
        coords_replaceable,
        selected_coords,
        title="Replacement Relationships: Optimized vs Replaceable Stations",
        title_fontsize=15,
        color='dodgerblue',
        save_path=output_figure_dodgerblue
    )

    # Compute and print coverage metrics
    covered_weights = sum(weights[i] for i in covered)
    total_weights = weights.sum()
    weighted_cov_ratio = covered_weights/total_weights if total_weights>0 else 0
    print(f"实际选中站点数: {len(selected)}/{MAX_STATIONS}")
    print(f"加权覆盖分数: {covered_weights:.3f}")
    print(f"权重覆盖率: {weighted_cov_ratio*100:.1f}%")
