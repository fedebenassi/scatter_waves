==== BEGIN README.md ====
# Wave Model Validation Toolkit

**Advanced validation toolkit for WAVEWATCH III (WW3) unstructured models against multi-satellite observations**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)](https://github.com/scausio/scatter_waves)

---

## 🌊 Overview

This toolkit provides comprehensive validation capabilities for wave model outputs (WAVEWATCH III) against multi-satellite observations. It features advanced statistical analysis, extreme event validation, and publication-quality visualizations.

### Key Features

- **📊 52 Comprehensive Metrics** - Standard and percentile-based validation metrics
- **🎯 Extreme Event Analysis** - Percentile-based validation (P75, P90, P95, P99)
- **📈 Advanced Visualizations** - Scatter plots, Q-Q plots, time series, spatial maps
- **🗺️ Multi-satellite Support** - CFOSAT, Cryosat-2, AltiKa, HY-2B, Jason-3, Sentinel-3A/B
- **🔧 Flexible Configuration** - YAML-based configuration system
- **🚀 Production Ready** - Optimized for HPC environments (CMCC Zeus)

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Conda/Mamba (recommended)
- Access to WAVEWATCH III model outputs
- Satellite altimetry data (L3 products)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/scausio/scatter_waves.git
cd scatter_waves

# Create conda environment
conda env create -f utils/environment.yml
conda activate wave-validation

# Or install dependencies manually
pip install xarray numpy matplotlib scipy scikit-learn seaborn pyyaml munch basemap
```

### Required Libraries

```yaml
dependencies:
  - python>=3.8
  - xarray
  - numpy
  - matplotlib>=3.5
  - scipy
  - scikit-learn
  - seaborn
  - pyyaml
  - munch
  - basemap
  - netCDF4
```

---

## 🚀 Quick Start

### 1. Configure Your Analysis

Edit `conf.yaml` with your paths and settings:

```yaml
sat_preproc:
  sat_names: ['CFOSAT', 'Jason-3', 'Sentinel-3A']
  years: [2023]
  paths:
    sat: '/path/to/satellite/data/{satName}/{year}/{month}/'
    output: '/path/to/output/'
  processing:
    boundingBox:
      xmin: 12.0
      xmax: 21.0
      ymin: 38.0
      ymax: 46.0

model_preproc:
  out_dir: '/path/to/output/'
  datasets:
    sat:
      path: '/{year}_ALLSAT_threshold_landMasked_qcheck_zscore3.nc'
    models:
      my_experiment:
        type: 'unstructured'
        path: '/path/to/model/ww3*.nc'
        lat: 'latitude'
        lon: 'longitude'
        hs: 'hs'
        time: 'time'

plot:
  title: 'My Validation Study'
  experiments:
    my_experiment:
      series: '/my_experiment_{year}_sat_series.nc'
  out_dir: '/path/to/plots/'
  filters:
    percentile_thresholds: [75, 90, 95]  # Extreme event analysis
    ntimes: 3
    min: 0.15
    max: 20.0
```

### 2. Run the Complete Workflow

#### Satellite Validation (Default)

```bash
# Submit complete satellite validation workflow
python main.py -c conf.yaml -s 20230101 -e 20230131

# Or run individual steps
python sat_preprocessing.py -c conf.yaml
python mergeSats.py -c conf.yaml
python model_preprocessingUnstruct.py -c conf.yaml
python validation.py -c conf.yaml
```

#### Buoy Validation (NEW)

```bash
# Run buoy validation workflow
python main.py --use-buoy -s 20210101 -e 20210131

# Arguments:
# --use-buoy : Use buoy data instead of satellite observations
# -s YYYYMMDD : Start date
# -e YYYYMMDD : End date
# -c conf.yaml : Configuration file (optional)
```

**Buoy Data Sources Supported:**
- **CSV files** - CMEMS or custom CSV format
- **Copernicus Marine Service API** - Direct API access
- **ISPRA folder structure** - Station folders with monthly CSV files
- **Nausicaa single file** - Consolidated multi-year observations

**Multi-source Processing:**
Process multiple buoy sources in a single run by configuring `sources` in conf.yaml:

```yaml
buoy_preproc:
  input:
    sources: ['csv', 'ispra_folders', 'copernicusmarine']  # Process all at once
```

### 3. View Results

Generated plots will be saved to your configured output directory:
- `validation_scatter_*.png` - Comprehensive scatter plots with statistics
- `validation_timeseries_*.png` - Time series analysis
- `validation_map_*.png` - Spatial validation maps
- `validation_tracks_*.png` - Satellite track analysis

---

## 📚 Module Documentation

### 1. `buoy_preprocessing.py` - Buoy Data Preprocessing (NEW)

Process buoy observations into model-compatible NetCDF format with support for multiple data sources.

#### Features

- **Multi-source support**: CSV files, Copernicus Marine API, ISPRA folders, Nausicaa
- **Parallel processing**: Process multiple sources in a single run
- **Quality control**: Variable-specific QC flags, geographic filtering, minimum observation thresholds
- **Multiple variables**: Significant wave height, period, direction
- **Timezone handling**: Automatic detection and conversion to UTC naive
- **Station-based organization**: Per-station NetCDF files + merged series file
- **Model compatibility**: Output format matches satellite preprocessing for unified validation workflow

#### Supported Data Sources

**1. CSV Files (CMEMS format)**
```yaml
buoy_preproc:
  input:
    sources: ['csv']
    path: '/path/to/cmems_buoy_data.csv'
  columns:
    time: 'TIME'
    lon: 'LONGITUDE'
    lat: 'LATITUDE'
    hs: 'VHM0'
    station: 'PLATFORM_CODE'
```

**2. Copernicus Marine Service API**
```yaml
buoy_preproc:
  input:
    sources: ['copernicusmarine']
    copernicusmarine:
      dataset_id: 'cmems_obs-ins_glo_wav_my_na_irr'
      variables: ['VHM0', 'VMDR', 'VTM10']
```

**3. ISPRA Folder Structure + Nausicaa**
```yaml
buoy_preproc:
  input:
    sources: ['ispra_folders']
    ispra_folders:
      base_path: '/path/to/ispra_buoys'
      nausicaa:
        enabled: true
        file_path: '/path/to/nausicaa_2007_2023.txt'
        position: [41.85, 17.18]
      station_positions:
        alghero: [40.548611, 8.106944]
        # ... more stations
```

**4. Multi-source Processing**
```yaml
buoy_preproc:
  input:
    sources: ['csv', 'ispra_folders']  # Process multiple sources!
```

#### Usage

```python
from buoy_preprocessing import submit as buoy_preproc

ds = buoy_preproc(conf_path='conf.yaml', start_date='20210101', end_date='20210131')
```

Output: Per-station folders + combined series file ready for model_preprocessing.

---

### 2. `stats.py` - Statistical Metrics Module

Comprehensive statistical functions for model validation.

#### Standard Metrics

```python
import stats

# Calculate basic metrics
bias = stats.BIAS(ds, 'model_hs', 'hs')
rmse = stats.RMSE(ds, 'model_hs', 'hs')
mae = stats.MAE(ds, 'model_hs', 'hs')
r = stats.correlation(ds, 'model_hs', 'hs')
si = stats.ScatterIndex(ds, 'model_hs', 'hs')
skill = stats.skill_score(ds, 'model_hs', 'hs')
slope = stats.symmetric_slope(ds, 'model_hs', 'hs')
```

#### Quantile-Based Metrics

```python
# Analyze performance at specific quantiles
q_bias = stats.quantile_bias(ds, 'model_hs', 'hs', quantiles=[0.25, 0.5, 0.75, 0.95])
q_skill = stats.quantile_skill_score(ds, 'model_hs', 'hs', quantiles=[0.5, 0.9, 0.95, 0.99])
```

#### Comprehensive Metrics Calculation

```python
# Calculate all metrics including percentile-based
all_metrics = stats.metrics(
    ds, 
    model_var='model_hs', 
    obs_var='hs',
    percentile_thresholds=[75, 90, 95]
)

print(f"Overall RMSE: {all_metrics['RMSE']:.3f}")
print(f"P95 RMSE: {all_metrics['RMSE_P95']:.3f}")  # RMSE for extreme events (>95th percentile)
```

#### Available Metrics (52 total)

**Standard Metrics (11):**
- `BIAS` - Mean bias (model - obs)
- `RMSE` - Root Mean Square Error
- `MAE` - Mean Absolute Error
- `NRMSE` - Normalized RMSE
- `NMAE` - Normalized MAE
- `NBIAS` - Normalized Bias
- `SI` - Scatter Index
- `R` - Pearson correlation coefficient
- `R2` - Coefficient of determination
- `SKILL` - Murphy's skill score
- `SLOPE` - Symmetric slope

**Percentile-Based Metrics (for each threshold: P75, P90, P95):**
- `BIAS_PXX`, `RMSE_PXX`, `MAE_PXX`, `NRMSE_PXX`, `NMAE_PXX`, `NBIAS_PXX`, `SI_PXX`

**Statistical Information:**
- `N` - Number of observations
- `MEAN_MODEL` - Mean model value
- `MEAN_OBS` - Mean observation value
- `STD_MODEL` - Model standard deviation
- `STD_OBS` - Observation standard deviation

---

### 2. `validation.py` - Scatter Plot Validation

Generate comprehensive validation scatter plots with statistical analysis.

#### Features

- **Multi-panel layout**: Main scatter + marginal histograms + Q-Q plot + statistics table
- **Density coloring**: Hexbin for large datasets (N > 10,000)
- **Regression analysis**: 1:1 line, best fit, confidence bands
- **Extreme event focus**: Separate validation for high percentiles
- **Publication quality**: 300 DPI, professional styling

#### Usage

```python
from validation import scatter_waves

# Create comprehensive scatter plot
scatter_waves(
    ds=dataset,
    model_var='model_hs',
    obs_var='hs',
    title='Model Validation - September 2023',
    outname='/path/to/output.png',
    maskNtimes=3,
    percentile_thresholds=[75, 90, 95]
)
```

#### Output

Creates a 4-panel figure:
1. **Main scatter plot** - Model vs observations with density coloring
2. **Marginal histograms** - Distribution comparison
3. **Q-Q plot** - Quantile-quantile analysis for distribution matching
4. **Statistics table** - 25+ metrics including percentile-based validation

---

### 3. `validation_timeseries.py` - Time Series Analysis

Temporal validation with rolling statistics and trend analysis.

#### Features

- **Three-plot system**:
  1. Main time series with confidence intervals
  2. Error metrics evolution (NRMSE, NBIAS, MAE)
  3. Extreme event analysis (percentile-based)
- **Daily statistics** with error bars
- **Rolling mean** for trend identification
- **Multi-model comparison**

#### Usage

```python
from validation_timeseries import timeseries

# Generate time series validation
timeseries(
    ds=dataset,
    title='Temporal Validation',
    outname='/path/to/timeseries.png',
    maskNtimes=3,
    percentile_thresholds=[75, 90, 95]
)
```

#### Advanced Options

```python
# Custom rolling window and confidence intervals
timeseries(
    ds=dataset,
    title='7-day Rolling Analysis',
    outname='timeseries_7d.png',
    maskNtimes=3,
    percentile_thresholds=[90, 95, 99],
    rolling_window=7,  # 7-day rolling mean
    confidence_level=0.95  # 95% confidence intervals
)
```

---

### 4. `validation_map.py` - Spatial Validation Maps

Generate spatial validation maps with geographic context.

#### Features

- **2D spatial binning** for gridded analysis
- **Multiple metrics**: BIAS, RMSE, NRMSE, NBIAS, MAE
- **Coastline overlay** with high-resolution basemap
- **Percentile-based maps** for extreme events
- **Statistics overlay** with mean, std, min, max

#### Usage

```python
from validation_map import plotMap

# Create spatial validation map
plotMap(
    ds=dataset,
    var='BIAS',
    title='Spatial Bias Analysis',
    outname='/path/to/map.png',
    coast_resolution='h',  # High resolution
    binning_res=0.05  # 0.05° grid
)
```

#### Available Map Types

```python
# Standard metrics
for metric in ['BIAS', 'RMSE', 'NRMSE', 'NBIAS', 'MAE']:
    plotMap(ds, var=metric, title=f'Spatial {metric}', outname=f'map_{metric}.png')

# Percentile-based metrics (extreme events)
for metric in ['BIAS_P90', 'RMSE_P90', 'RMSE_P95']:
    plotMap(ds, var=metric, title=f'Extreme Event {metric}', outname=f'map_{metric}.png')
```

---

### 5. `validation_tracks.py` - Satellite Track Analysis

Visualize satellite track validation with spatial context.

#### Features

- **Daily track analysis** for temporal evolution
- **Metric-based coloring** (BIAS, RMSE, NRMSE, NBIAS)
- **High-resolution coastline**
- **Statistics overlay**

#### Usage

```python
from validation_tracks import plotTracks

# Generate satellite track plots
plotTracks(
    ds=dataset,
    var='bias',
    title='Satellite Track Validation',
    outname='/path/to/tracks.png',
    maskNtimes=3,
    daily=True  # Create separate plots for each day
)
```

---

## 🎯 Advanced Features

### Extreme Event Validation

Focus validation on high-impact events using percentile thresholds:

```yaml
# In conf.yaml
plot:
  filters:
    percentile_thresholds: [75, 90, 95, 99]  # Top 25%, 10%, 5%, 1%
```

This generates additional metrics:
- `RMSE_P75` - RMSE for waves above 75th percentile
- `BIAS_P90` - Bias for waves above 90th percentile
- `SI_P95` - Scatter index for waves above 95th percentile
- `MAE_P99` - MAE for waves above 99th percentile (extreme events)

### Quantile Analysis

Analyze model performance across the distribution:

```python
# Q-Q plot automatically generated in scatter plots
# Compare quantiles at [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

# Or use quantile metrics directly
quantile_bias = stats.quantile_bias(ds, 'model_hs', 'hs', quantiles=[0.5, 0.9, 0.95, 0.99])
quantile_skill = stats.quantile_skill_score(ds, 'model_hs', 'hs', quantiles=[0.9, 0.95, 0.99])
```

### Multi-Model Comparison

Compare multiple model configurations:

```yaml
model_preproc:
  datasets:
    models:
      baseline:
        path: '/path/to/baseline/ww3*.nc'
      improved:
        path: '/path/to/improved/ww3*.nc'
      experimental:
        path: '/path/to/experimental/ww3*.nc'

plot:
  experiments:
    baseline:
      series: '/baseline_{year}_sat_series.nc'
    improved:
      series: '/improved_{year}_sat_series.nc'
    experimental:
      series: '/experimental_{year}_sat_series.nc'
```

### Custom Filtering

Apply sophisticated data filters:

```yaml
sat_preproc:
  filters:
    quality_check:
      variable_name: 'quality_flag'
      value: 1  # Good data only
    land_masking:
      variable_name: 'surface_type'
      value: 0  # Ocean only
      shapefile: '/path/to/ocean_mask.shp'
    threshold:
      min: 0.25  # Minimum Hs (m)
      max: 20.0  # Maximum Hs (m)
    zscore:
      sigma: 3  # Remove outliers beyond 3σ

model_preproc:
  filters:
    max_distance_in_space: 0.01  # degrees (~1 km)
    max_distance_in_time: 1  # hours
    threshold:
      min: 0.15
      max: 20.0

plot:
  filters:
    ntimes: 3  # Mask if sat_hs > ntimes * model_hs
    min: 0.15
    max: 20.0
```

---

## 📊 Output Files

### Generated Files

```
output/
├── satellite_preprocessing/
│   ├── CFOSAT_2023.nc
│   ├── Jason-3_2023.nc
│   └── Sentinel-3A_2023.nc
├── merged/
│   └── 2023_ALLSAT_threshold_landMasked_qcheck_zscore3.nc
├── model_series/
│   └── my_experiment_2023_sat_series.nc
└── plots/
    ├── validation_scatter_my_experiment.png
    ├── validation_timeseries_my_experiment.png
    ├── validation_map_BIAS_my_experiment.png
    ├── validation_map_RMSE_my_experiment.png
    ├── validation_map_BIAS_P90_my_experiment.png  # Extreme events
    └── validation_tracks_my_experiment.png
```

### File Descriptions

**Preprocessed Data:**
- `*_ALLSAT_*.nc` - Merged satellite data with quality filters
- `*_sat_series.nc` - Model-satellite matched time series

**Validation Plots:**
- `validation_scatter_*.png` - 4-panel scatter analysis (1200x1000 px, 300 DPI)
- `validation_timeseries_*.png` - 3-panel temporal analysis (1800x1200 px, 300 DPI)
- `validation_map_*.png` - Spatial validation maps (1200x1000 px, 300 DPI)
- `validation_tracks_*.png` - Satellite track analysis (1200x1000 px, 300 DPI)

---

## 🔧 Configuration Reference

### Complete Configuration Example

```yaml
# ============================================================================
# SATELLITE PREPROCESSING
# ============================================================================
sat_preproc:
  sat_names: ['CFOSAT', 'Cryosat-2', 'AltiKa', 'HY-2B', 'Jason-3', 'Sentinel-3A', 'Sentinel-3B']
  years: [2023]
  
  sat_specifics:
    lat: 'latitude'
    lon: 'longitude'
    time: 'time'
    hs: 'VAVH'
    type: 'L3'
  
  paths:
    sat: '/data/satellite/{satName}/{year}/{month}/'
    output: '/work/validation/output/'
  
  filenames:
    sat:
      template: 'global_vavh_l3_rt_*_*.nc'
      output: '{sat_name}_{year}'
  
  processing:
    boundingBox:
      xmin: 12.0
      xmax: 21.0
      ymin: 38.0
      ymax: 46.0
  
  filters:
    quality_check:
      variable_name: 'quality_flag'
      value: 1
    land_masking:
      variable_name: 'surface_type'
      value: 0
      shapefile: '/path/to/ocean_mask.shp'
    threshold:
      min: 0.25
      max: 20.0
    zscore:
      sigma: 3

# ============================================================================
# MODEL PREPROCESSING
# ============================================================================
model_preproc:
  out_dir: '/work/validation/output/'
  
  datasets:
    sat:
      path: '/{year}_ALLSAT_threshold_landMasked_qcheck_zscore{sigma}.nc'
    
    models:
      baseline:
        type: 'unstructured'
        path: '/work/model/baseline/ww3*.nc'
        lat: 'latitude'
        lon: 'longitude'
        hs: 'hs'
        time: 'time'
      
      improved:
        type: 'unstructured'
        path: '/work/model/improved/ww3*.nc'
        lat: 'latitude'
        lon: 'longitude'
        hs: 'hs'
        time: 'time'
  
  filters:
    max_distance_in_space: 0.01  # degrees
    max_distance_in_time: 1      # hours
    threshold:
      min: 0.15
      max: 20.0

# ============================================================================
# PLOTTING & VALIDATION
# ============================================================================
plot:
  title: 'WAVEWATCH III Validation - 2023'
  coast_resolution: 'i'  # 'c'=crude, 'l'=low, 'i'=intermediate, 'h'=high
  binning_res: 0.05      # degrees for spatial binning
  
  experiments:
    baseline:
      series: '/baseline_{year}_sat_series.nc'
    improved:
      series: '/improved_{year}_sat_series.nc'
  
  out_dir: '/work/validation/plots/'
  
  filters:
    percentile_thresholds: [75, 90, 95, 99]  # Extreme event analysis
    ntimes: 3                                  # Outlier detection
    min: 0.15                                  # Minimum Hs (m)
    max: 20.0                                  # Maximum Hs (m)
```

---

## 💡 Usage Examples

### Example 1: Regional Validation Study

```bash
# 1. Configure for Mediterranean Sea
cat > conf_mediterranean.yaml << EOF
sat_preproc:
  sat_names: ['Jason-3', 'Sentinel-3A', 'Sentinel-3B']
  years: [2023]
  processing:
    boundingBox:
      xmin: 0.0
      xmax: 36.0
      ymin: 30.0
      ymax: 46.0

plot:
  title: 'Mediterranean Sea Validation 2023'
  filters:
    percentile_thresholds: [90, 95]
EOF

# 2. Run analysis
python main.py -c conf_mediterranean.yaml
```

### Example 2: Extreme Event Focus

```bash
# Configure for storm validation
cat > conf_storms.yaml << EOF
plot:
  title: 'Storm Event Validation'
  filters:
    percentile_thresholds: [95, 99]  # Focus on top 5% and 1%
    min: 3.0  # Only waves > 3m
    max: 20.0
EOF

python validation.py -c conf_storms.yaml
```

### Example 3: Multi-Model Comparison

```python
# Python script for batch processing
import subprocess

models = ['ST2', 'ST4', 'ST6']

for model in models:
    config = f'conf_{model}.yaml'
    subprocess.run(['python', 'main.py', '-c', config])
    print(f"Completed validation for {model}")
```

### Example 4: HPC Batch Submission

```bash
#!/bin/bash
#BSUB -P R000
#BSUB -J wave_validation
#BSUB -n 1
#BSUB -W 02:00
#BSUB -o validation_%J.out
#BSUB -e validation_%J.err

module load conda
conda activate wave-validation

python main.py -c conf.yaml
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `KeyError: 'hs'`
```bash
# Solution: Check variable names in configuration
# Ensure model/satellite variable names match your data
```

**Issue**: Empty plots or no data
```bash
# Solution: Check spatial/temporal matching
# Verify boundingBox and max_distance_in_space/time settings
```

**Issue**: Memory errors with large datasets
```bash
# Solution: Process data in chunks or reduce spatial resolution
# Increase binning_res or reduce time range
```

**Issue**: Missing percentile metrics
```bash
# Solution: Ensure percentile_thresholds is set in conf.yaml
plot:
  filters:
    percentile_thresholds: [75, 90, 95]
```

### Performance Optimization

```yaml
# For large datasets, optimize processing:
sat_preproc:
  processing:
    chunk_size: 1000000  # Process in chunks
    parallel: True       # Enable parallel processing

model_preproc:
  filters:
    max_distance_in_space: 0.01  # Tighter matching reduces data volume
```

---

## 📖 Workflow Details

### Complete Processing Pipeline

```
┌─────────────────────┐
│  Satellite Data     │
│  (Multi-mission)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ sat_preprocessing.py│  ← Quality filters, land masking, Z-score
│  (per satellite)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   mergeSats.py      │  ← Combine all satellites
│  (annual files)     │
└──────────┬──────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Model Output      │  │  Merged Satellite   │
│   (WW3 unstr)       │  │                     │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           └───────────┬────────────┘
                       ▼
           ┌─────────────────────────┐
           │model_preprocessingUnstr │  ← Spatial/temporal matching
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │   validation.py         │  ← Generate all plots
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │  Validation Products    │
           │  • Scatter plots        │
           │  • Time series          │
           │  • Spatial maps         │
           │  • Satellite tracks     │
           └─────────────────────────┘
```

### Processing Steps Detail

1. **Satellite Preprocessing** (`sat_preprocessing.py`)
   - Read multi-mission satellite data
   - Apply quality checks
   - Apply land masking (dataset flag or shapefile)
   - Apply Z-score filter for outliers
   - Subset by bounding box
   - Save annual files per satellite

2. **Satellite Merging** (`mergeSats.py`)
   - Combine all satellite missions
   - Merge annual datasets
   - Ensure consistent formatting

3. **Model Preprocessing** (`model_preprocessingUnstruct.py`)
   - Read WW3 unstructured outputs
   - Match model-satellite pairs (space/time)
   - Apply distance filters
   - Create matched time series

4. **Validation & Plotting** (`validation.py`, `validation_timeseries.py`, etc.)
   - Calculate comprehensive metrics
   - Generate publication-quality plots
   - Save validation statistics

---

## 🔬 Scientific Background

### Validation Metrics

**Standard Metrics:**

$$BIAS = \frac{1}{N}\sum_{i=1}^{N}(M_i - O_i)$$

$$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(M_i - O_i)^2}$$

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|M_i - O_i|$$

$$SI = \frac{RMSE}{\bar{O}}$$

$$R = \frac{\sum(M_i - \bar{M})(O_i - \bar{O})}{\sqrt{\sum(M_i - \bar{M})^2\sum(O_i - \bar{O})^2}}$$

**Percentile-Based Metrics:**

For data above percentile threshold $P_{xx}$:

$$RMSE_{P_{xx}} = \sqrt{\frac{1}{N_{P_{xx}}}\sum_{O_i > P_{xx}}(M_i - O_i)^2}$$

**Quantile Analysis:**

Compare model and observation quantiles:

$$Q_{skill} = 1 - \frac{\sum(Q_M(p) - Q_O(p))^2}{\sum(Q_O(p) - \bar{O})^2}$$

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-metric`)
3. **Commit** changes (`git commit -m 'Add new metric'`)
4. **Push** to branch (`git push origin feature/new-metric`)
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/scausio/scatter_waves.git
cd scatter_waves
conda env create -f utils/environment.yml
conda activate wave-validation
```

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Add tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/scausio/scatter_waves/issues)
- **Email**: scausio@cmcc.it
- **Documentation**: [GitHub Wiki](https://github.com/scausio/scatter_waves/wiki)

---

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{scatter_waves_2024,
  title = {Wave Model Validation Toolkit},
  author = {Causio, S.},
  year = {2024},
  url = {https://github.com/scausio/scatter_waves},
  version = {2.0.0}
}
```

---

## 🏆 Acknowledgments

- **CMCC Foundation** - Centro Euro-Mediterraneo sui Cambiamenti Climatici
- **Copernicus Marine Service** - Satellite data provider
- **WAVEWATCH III** - Wave model development team

---

## 📈 Version History

### Version 2.0.0 (2024-12)
- ✨ Added 52 comprehensive metrics (11 standard + percentile variants)
- 🎯 Implemented percentile-based validation for extreme events
- 📊 Enhanced visualizations (Q-Q plots, confidence intervals)
- 📚 Comprehensive documentation with detailed examples
- 🚀 Publication-quality graphics (300 DPI)
- 🔧 Improved code quality (docstrings, type hints, error handling)

### Version 1.0.0 (Initial)
- Basic validation functionality
- Standard metrics (BIAS, RMSE, SI)
- Basic plotting capabilities

---

## 🔗 Related Projects

- [WAVEWATCH III](https://github.com/NOAA-EMC/WW3)
- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [xarray](http://xarray.pydata.org/)

---

*For questions, suggestions, or collaborations, please open an issue or contact the maintainers.*

==== END README.md ====