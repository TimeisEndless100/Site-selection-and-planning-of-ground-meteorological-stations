# RMCLP: Replaceable Maximum Coverage Location Problem

Optimize meteorological station placement by maximizing spatial coverage through strategic substitution of redundant stations.

## Introduction

Meteorological observations are essential for accurate weather forecasting, climate modeling, and resource management. However, establishing and maintaining ground-based stations involves significant costs. The Replaceable Maximum Coverage Location Problem (RMCLP) provides a systematic, data-driven framework to optimize station networks by integrating new candidate sites with existing infrastructure and enabling replacement of redundant stations, thereby maximizing coverage efficiency under resource constraints.

## Methodology

1. **Data Preprocessing**: Harmonize multi-source rasters (landcover, slope, DEM) to a common CRS, clip to the study region, and extract fixed-size image patches labeled by station presence.
2. **Feature Extraction & Inference**: Use a lightweight CNN to predict station occurrence probabilities on unlabeled patches, informing spatial demand distribution.
3. **Spatial Analyses**:
   - **Density Classification**: Categorize existing stations as sparse or dense via nearest-neighbor distances.
   - **Gradient Centroid Extraction**: Identify high-variability rainfall regions through gradient thresholding and centroid computation.
   - **Precipitation Variability Mapping**: Compute coefficient of variation (CV) from multi-year rainfall rasters to weight demand points by climatic volatility.
4. **Greedy RMCLP Solver**: Iteratively select station locations that maximize weighted coverage within a service radius, substituting replaceable existing stations when candidate sites offer higher marginal gains.

## Repository Structure

```
├── Data/                                    # Input data (rasters, boundary, station CSV)
├── patches--64/                              # Extracted patches and metadata
├── requirements.txt                         # Python dependencies
├── 1.Data_Preprocessing.py                   # Data alignment and patch generation
├── 2.cnn.py                                  # CNN model training and inference
├── 3.Visualization_of_Predicted_Probability_Maps.py
├── 4.Dynamic_Decay_Strategy.py
├── 5.Identification_of_Sparse_and_Dense_Areas.py
├── 6.Extraction_of_Rainfall_Slope_Map_from_Rainfall_Raster_Data.py
├── 7.Centroid_Extraction_of_Connected_Regions_in_Rainfall_Gradient_Fields.py
├── 8.calculate_precipitation_cv.py
├── 9.greedy_maximum_coverage_solver.py      # RMCLP implementation
└── README.md                                 # Project documentation
```

## Environment Setup

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Script Usage

### 1. Data Preprocessing (`1.Data_Preprocessing.py`)
Extract unified rasters and labeled patches:

```bash
python 1.Data_Preprocessing.py --landcover Data/land_cover_classification_30m.tif \
    --slope Data/重庆_slope_30m.tif --dem Data/DEM_30m分辨率.tif \
    --stations Data/Cq_positive_case.csv --boundary Data/重庆市_市.geojson \
    --out_tif unified_3band.tif --patch_dir patches--64 --patch_size 64
```

Outputs:
- `unified_3band.tif`
- `patches--64/patches_info.csv`
- Patch TIFFs and `patch_label_distribution.png`

### 2. CNN Training & Inference (`2.cnn.py`)
Train and predict patch-based probabilities:

```bash
python 2.cnn.py --mode train --csv patches--64/patches_info.csv --patch_dir patches--64 \
    --epochs 50 --batch_size 64 --output_model best_model.pth
python 2.cnn.py --mode infer --model best_model.pth --csv patches--64/patches_info.csv \
    --patch_dir patches--64 --output_csv patches_predictions.csv
```

Outputs:
- `best_model.pth`
- `patches_predictions.csv`

### 3. Predicted Probability Visualization (`3.Visualization_of_Predicted_Probability_Maps.py`)
Plot probability map:

```bash
python "3.Visualization of Predicted Probability Maps.py" \
    --coords_csv patches--64/patches_64_with_coords.csv \
    --pred_csv patches_predictions.csv --output_png prediction_probability_map.png
```

### 4. Dynamic Decay Strategy (`4.Dynamic_Decay_Strategy.py`)
Apply distance-based decay:

```bash
python 4.Dynamic_Decay_Strategy.py \
    --candidates patches--64/patches_64_with_coords.csv \
    --preds patches_predictions.csv --stations Data/Cq_positive_case.csv \
    --output_png adjusted_prediction_probability_map.png
```

### 5. Station Density Classification (`5.Identification_of_Sparse_and_Dense_Areas.py`)
Classify and map station densities:

```bash
python 5.Identification_of_Sparse_and_Dense_Areas.py \
    --stations Data/Cq_positive_case.csv --boundary Data/重庆市_市.geojson \
    --output_png station_density_classification_map.png
```

### 6. Rainfall Slope Map Extraction (`6.Extraction_of_Rainfall_Slope_Map_from_Rainfall_Raster_Data.py`)
Compute rainfall gradients:

```bash
python 6.Extraction_of_Rainfall_Slope_Map_from_Rainfall_Raster_Data.py \
    --input_tif Data/2022重庆_降雨量2022.tif \
    --output_png Rainfall_Gradient_Sobel_Coordinates.png \
    --output_tif Rainfall_Gradient_Sobel.tif
```

### 7. High-Gradient Centroid Extraction (`7.Centroid_Extraction_of_Connected_Regions_in_Rainfall_Gradient_Fields.py`)
Extract centroids of high-gradient zones:

```bash
python 7.Centroid_Extraction_of_Connected_Regions_in_Rainfall_Gradient_Fields.py \
    --input_tif Rainfall_Gradient_Sobel.tif --threshold 25 \
    --output_csv High_Rainfall_Gradient_Points_gt25.csv \
    --output_png High_Rainfall_Gradient_Points_Map_gt25_markerX.png
```

### 8. Precipitation CV Computation (`8.calculate_precipitation_cv.py`)
Calculate coefficient of variation:

```bash
python 8.calculate_precipitation_cv.py --input_files Data/2019重庆_降雨量.tif \
    Data/2020重庆_降雨量.tif Data/2021重庆_降雨量.tif Data/2022重庆_降雨量2022.tif \
    --output_tif rainfall_cv.tif
```

### 9. RMCLP Optimization (`9.greedy_maximum_coverage_solver.py`)
Solve the station placement problem:

```bash
python 9.greedy_maximum_coverage_solver.py \
    --candidates Data/high_prob_candidates.csv --existing Data/Cq_positive_case.csv \
    --demand Data/Combined_Gradient_gt25_gt50_AllPoints.csv --cv_raster rainfall_cv.tif \
    --boundary Data/重庆市_市.geojson --output_selected selected_optimized_stations.csv \
    --output_replace replacement_mapping.csv \
    --output_png replacement_visualizationstations_custom_legend.png
```

## Data Preparation

1. Place all input rasters and GeoJSON in `Data/` directory.
2. Confirm file paths in `1.Data_Preprocessing.py`.
3. Ensure `patches--64/` directory exists or let the script create it.

## License

This project is licensed under the MIT License.

