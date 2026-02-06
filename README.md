# Philippine Earthquake Clustering Analysis

A comprehensive machine learning pipeline for analyzing and clustering Philippine earthquake data using location-agnostic features. This project includes automated data scraping, feature engineering, clustering analysis, and an interactive Streamlit web application for real-time earthquake monitoring.

## Key Features

### 1. **Deployed Streamlit App** 

An interactive web application (`app/entrypoint.py`) that provides real-time earthquake cluster monitoring:

- **Live Data Integration**: Automatically scrapes the latest earthquake data from PHIVOLCS
- **HDBSCAN Clustering**: Real-time cluster prediction using trained models
- **Interactive Map Visualization**: Philippine map overlay showing clustered seismic events with distinct color coding
- **Multi-Version Model Support**: Load and compare different model versions
- **Cluster Interpretation**: Displays cluster labels and descriptions from model metadata
- **Adjustable Display**: Customize the number of events displayed (5-100)

**Run the app:**
```bash
streamlit run app/entrypoint.py
```

### 2. **Automated Data Scraper** 

The `scraper.py` module fetches earthquake data directly from PHIVOLCS (Philippine Institute of Volcanology and Seismology):

- Scrapes historical monthly data from 2018-2025
- Parses HTML tables to extract earthquake records
- Handles SSL certificate issues with automatic fallback
- Exports data to CSV format with standardized columns (datetime, latitude, longitude, depth, magnitude, location)

**Usage:**
```python
from scraper import scrape_earthquake_data
df = scrape_earthquake_data()
```

### 3. **Comprehensive Clustering Analysis** 🔬

The `clustering_experiment.py` script performs extensive clustering analysis on earthquake data:

- **Location-Agnostic Features**: Clustering based on seismic characteristics (magnitude, depth, temporal patterns) rather than geographic proximity
- **Multiple Algorithms**: K-Means, DBSCAN, and HDBSCAN with automated hyperparameter tuning via grid search
- **Feature Engineering**: 73 engineered features including temporal patterns, local density, interevent times, and rolling statistics
- **Dimensionality Reduction**: PCA for optimal feature selection
- **Comprehensive Evaluation**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index

## Results from 100k Datapoints Experiment

Based on the analysis of **90,042 earthquake records** (M ≥ 2.0) from `clustering_experiment_results_100k_datapoints/`:

### Dataset Overview
- **Records**: 90,042 earthquakes
- **Time Period**: Historical data from PHIVOLCS
- **Features**: 10 location-agnostic features (PCA-reduced from 73 engineered features)
- **Explained Variance**: 95.0% retained after dimensionality reduction

### Key Findings

#### Clustering Performance

| Algorithm | Clusters | Noise | Silhouette Score | Key Driver |
|-----------|----------|-------|------------------|------------|
| **HDBSCAN** | 2 | 0.4% | **0.770** | Depth (binary split) |
| DBSCAN | 3 | 9.6% | 0.413 | Depth (extreme outliers) |
| K-Means | 3 | 0% | 0.244 | Magnitude + Depth |

**HDBSCAN achieved the best clustering quality** with excellent separation (Silhouette = 0.770) and minimal noise (0.4%).

#### Feature Importance

The analysis revealed that **depth is the single most important feature** across all algorithms:

1. **Depth** (0.688) - Primary separator between shallow and deep earthquakes
2. **Depth_diff_from_prev** (0.576) - Depth transitions between events
3. **Energy_concentration** (0.091) - Relative energy dominance
4. **Local_density** (0.068) - Spatial earthquake density

#### Cluster Interpretations

**HDBSCAN discovered 2 distinct earthquake behavior types:**

1. **Deep Anomalies** (0.07%): Ultra-deep earthquakes (500-700+ km) with extreme depth separation
2. **Background Seismicity** (99.6%): Main population including crustal and intermediate-depth events

**K-Means identified 3 clusters:**
1. Background Seismicity (66.3%) - Small, shallow routine events
2. Significant Shallow Events (24.4%) - Moderate-to-strong shallow earthquakes
3. Deep Subduction Events (9.2%) - Deep-focus subduction zone earthquakes

### Visualizations

The experiment generated comprehensive visualizations:

- **Philippine Map Comparisons**: Spatial distribution of clusters across the archipelago
- **Temporal Profiles**: Time-based patterns for each cluster
- **Cluster Statistics**: Magnitude, depth, and temporal characteristics
- **Feature Distributions**: Histograms of engineered features
- **Feature Importance**: Cross-algorithm consensus rankings
- **Grid Search Results**: Hyperparameter tuning visualizations
- **Algorithm Comparison**: Side-by-side cluster quality metrics

All visualizations are available in `clustering_experiment_results_100k_datapoints/`.

## Project Structure

```
earthquake-model/
├── app/                          # Streamlit web application
│   ├── entrypoint.py            # Main Streamlit app
│   ├── components/              # Modular components
│   │   ├── scraper.py          # Live data scraping
│   │   ├── feature_engineer.py # Feature engineering pipeline
│   │   ├── preprocess.py       # Preprocessing utilities
│   │   └── model_loader.py     # Model loading and management
│   └── models/                  # Trained model versions
│       ├── v1/                 # Model version 1
│       └── v2/                 # Model version 2 (with cluster labels)
│
├── clustering_experiment.py     # Main clustering analysis script
├── scraper.py                   # Standalone scraper module
├── data/                        # Earthquake data CSVs
│
├── clustering_experiment_results_100k_datapoints/  # Analysis results
│   ├── ANALYSIS.md             # Detailed analysis report
│   ├── CLUSTERS.md             # Cluster interpretation guide
│   ├── *.png                   # Visualization outputs
│   └── *.csv                   # Metrics and cluster data
│
└── requirements.txt            # Python dependencies
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd earthquake-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit` - Web application framework
- `hdbscan` - Hierarchical density-based clustering
- `scikit-learn` - Machine learning utilities
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `pydeck` - Interactive map visualization
- `beautifulsoup4`, `requests` - Web scraping

## Usage

### Running the Clustering Experiment

```bash
python clustering_experiment.py
```

The script will:
1. Load earthquake data from CSV
2. Engineer location-agnostic features
3. Perform dimensionality reduction (PCA)
4. Run grid search for hyperparameter tuning
5. Evaluate clustering quality
6. Generate visualizations and save results

### Running the Streamlit App

```bash
streamlit run app/entrypoint.py
```

The app will:
1. Load available model versions from `app/models/`
2. Automatically scrape latest data from PHIVOLCS
3. Display clustered events on an interactive Philippine map
4. Show cluster interpretations and statistics

### Scraping Data

```python
from scraper import scrape_earthquake_data

# Scrape all historical data
df = scrape_earthquake_data()
df.to_csv('data/earthquake_data.csv', index=False)
```

## Methodology

### Location-Agnostic Clustering

Unlike traditional spatial clustering, this project uses **location-agnostic features** to discover patterns based on seismic characteristics:

- **Magnitude features**: Raw magnitude, z-scores, rolling statistics
- **Depth features**: Raw depth, depth categories, depth transitions
- **Temporal features**: Hour, day of week, interevent times, burstiness
- **Spatial-temporal**: Local density, events in time windows (without using lat/lng directly)

This approach reveals patterns that transcend geographic boundaries, such as:
- Depth-based stratification (shallow vs. deep earthquakes)
- Magnitude-based groupings (background vs. significant events)
- Temporal patterns (aftershock sequences, swarms)

### Feature Engineering Pipeline

1. **Temporal Features**: Hour, day of week, day of year, cyclical encodings
2. **Energy Features**: Log energy, energy ratios
3. **Rolling Statistics**: Moving averages, standard deviations over windows
4. **Depth Features**: Categorical bins, z-scores, depth transitions
5. **Magnitude Anomalies**: Z-scores, percentiles, significance flags
6. **Local Density**: Neighbor counts within radius
7. **Interevent Features**: Time since previous event, burstiness metrics

### Hyperparameter Tuning

Automated grid search optimizes:
- **K-Means**: Number of clusters (k), initialization runs
- **DBSCAN**: Epsilon (eps), minimum samples
- **HDBSCAN**: Minimum cluster size, minimum samples, cluster selection method

Optimization metric: `combined_with_penalty` (balances silhouette score, Calinski-Harabasz, Davies-Bouldin, and noise penalty)

## Key Insights

1. **Depth is the dominant separator**: All algorithms identify depth as the primary clustering dimension, separating shallow crustal events from deep subduction zone earthquakes.

2. **HDBSCAN performs best**: Achieves highest silhouette score (0.770) with minimal noise (0.4%), finding a clear binary split in the data.

3. **Temporal patterns are underutilized**: Features designed to capture aftershock sequences and temporal clustering show low importance, suggesting these patterns may require specialized spatiotemporal methods.

4. **Geographic patterns emerge indirectly**: While location coordinates are excluded, spatial patterns (local density, regional activity) still influence clustering through derived features.

## References

- **Data Source**: [PHIVOLCS](https://earthquake.phivolcs.dost.gov.ph/) - Philippine Institute of Volcanology and Seismology
- **Clustering Algorithms**: HDBSCAN, DBSCAN, K-Means via scikit-learn
- **Analysis Results**: See `clustering_experiment_results_100k_datapoints/ANALYSIS.md` and `CLUSTERS.md` for detailed findings

## Author

Maxell Milay

## License

[Add your license here]

---

For detailed analysis results, see:
- `clustering_experiment_results_100k_datapoints/ANALYSIS.md` - Comprehensive analysis report
- `clustering_experiment_results_100k_datapoints/CLUSTERS.md` - Cluster interpretation guide
