# MRAM-NetOpt: Meteorological Station Network Optimization Toolkit

## Overview
This repository provides a suite of Python scripts for preprocessing remote sensing data, training and inference of a lightweight CNN, and optimizing meteorological station deployment in Chongqing, China. The end-to-end workflow covers geospatial data preparation, patch extraction, model training/inference under positive-unlabeled settings, spatial visualization, density analysis, gradient computation, and a greedy maximum coverage algorithm (MRAM-NetOpt).

## Repository Structure
```
├── Data/                                    # Input data folder
│   ├── land cover classification_30m.tif     # Landcover raster
│   ├── 重庆_slope_30m.tif                    # Slope raster
│   ├── DEM_30m分辨率.tif                     # DEM raster
│   ├── 重庆市_市.geojson                     # Chongqing boundary
│   └── Cq_positive_case.csv                  # Station location CSV
│──patches--64/                          # Patch folder
│  └── ...                               # Generated patch TIFFs and CSVs
├── 1.Data_Preprocessing.py                 # Geospatial raster reprojection, clipping, patch extraction
├── 2.cnn.py                                 # Lightweight CNN training & inference (PU learning)
├── 3.Visualization_of_Predicted_Probability_Maps.py
├── 4.Dynamic_Decay_Strategy.py
├── 5.Identification_of_Sparse_and_Dense_Areas.py
├── 6.Extraction_of_Rainfall_Slope_Map_from_Rainfall_Raster_Data.py
├── 7.Centroid_Extraction_of_Connected_Regions_in_Rainfall_Gradient_Fields.py
├── 8.calculate_precipitation_cv.py
├── 9.greedy_maximum_coverage_solver.py
└── README.md
```

## Dependencies
- Python 3.7+
- numpy, pandas, geopandas, rasterio, shapely, pyproj, fiona
- torch, torchvision, scikit-learn
- matplotlib, tqdm, scikit-image, sympy

Install via conda:
```bash
conda create -n mram-netopt python=3.8 \
    numpy pandas geopandas rasterio shapely pyproj fiona matplotlib tqdm scikit-image sympy pytorch torchvision scikit-learn -c conda-forge -c pytorch
conda activate mram-netopt
```

## Script Usage

### 1. Data Preprocessing (1.Data_Preprocessing.py)
Prepare unified 3-band rasters (landcover, slope, DEM), clip to Chongqing boundary, extract labeled patches.
```bash
python 1.Data_Preprocessing.py
```
Generates:
- `unified_3band.tif` (merged raster)
- `patches--64/patches_info.csv` (metadata)
- Patch TIFF files and `patch_label_distribution.png`.

### 2. CNN Training & Inference (2.cnn.py)
Train a three-layer lightweight CNN on extracted patches and perform PU inference.
```bash
python 2.cnn.py
```
Outputs:
- `best_bce_model.pth` (trained weights)
- `patches_predictions.csv` (filename, probability, label)

### 3. Predicted Probability Visualization
```bash
python "3.Visualization of Predicted Probability Maps.py"
```
Produces `prediction_probability_map.png`.

### 4. Dynamic Decay Strategy
```bash
python 4.Dynamic_Decay_Strategy.py
```
Outputs `adjusted_prediction_probability_map.png`.

### 5. Station Density Classification
```bash
python 5.Identification_of_Sparse_and_Dense_Areas.py
```
Generates `station_density_classification_map.png`.

### 6. Rainfall Slope Map Extraction
```bash
python 6.Extraction_of_Rainfall_Slope_Map_from_Rainfall_Raster_Data.py
```
Produces `Rainfall_Gradient_Sobel_Coordinates.png` and `Rainfall_Gradient_Sobel.tif`.

### 7. Centroid Extraction of High-Gradient Regions
```bash
python 7.Centroid_Extraction_of_Connected_Regions_in_Rainfall_Gradient_Fields.py
```
Generates `High_Rainfall_Gradient_Points_gt25.csv` and visualization PNG.

### 8. Precipitation CV Computation
```bash
python 8.calculate_precipitation_cv.py
```
Outputs `rainfall_cv.tif` and STD/CV comparison figures.

### 9. Greedy Maximum Coverage Optimization (MRAM-NetOpt)
```bash
python 9.greedy_maximum_coverage_solver.py
```
Selects stations and saves:
- `selected_optimized_stations.csv`
- `replacement_mapping.csv`
- Replacement visualization PNG

## Data Preparation
1. Place all input TIFFs and GeoJSON in `Data/`.
2. Ensure `1.Data_Preprocessing.py` references correct file names.
3. Create an empty `patches--64/` directory or let script create it.