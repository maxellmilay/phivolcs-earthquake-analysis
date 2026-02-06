# Philippine Earthquake Clustering Analysis

A comprehensive machine learning pipeline for analyzing and clustering Philippine earthquake data using location-agnostic features. This project includes automated data scraping, feature engineering, clustering analysis, and an interactive Streamlit web application for real-time earthquake monitoring.

## 🎯 Key Features

### 1. **Deployed Streamlit App** 🌐

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

### 2. **Automated Data Scraper** 📡

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
- **Dimensionality Reduction**: PCA for optimal feature selection (15 features → 10 components, 95.3% variance retained)
- **Comprehensive Evaluation**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index

## 📊 Results from 100k Datapoints Experiment

Based on the analysis of **90,042 earthquake records** (M ≥ 2.0) from `clustering_results_100k_datapoints/`:

### Dataset Overview
- **Records**: 90,042 earthquakes
- **Time Period**: June 2018 - December 2025
- **Features**: 15 location-agnostic features (PCA-reduced to 10 components)
- **Explained Variance**: 95.3% retained after dimensionality reduction
- **Total Runtime**: 16.4 minutes (982.97 seconds)

### Key Findings

#### Clustering Performance

| Algorithm | Clusters | Noise | Silhouette Score | Combined Score |
|-----------|----------|-------|------------------|----------------|
| **HDBSCAN** | 4 | 0% | **0.725** | **0.853** |
| K-Means | 5 | 0% | 0.283 | 0.715 |
| DBSCAN | 9 | 7.6% | 0.179 | 0.610 |

**Winner: HDBSCAN** achieved the highest combined score (0.853) with excellent silhouette coefficient (0.725), indicating well-separated clusters based on seismic characteristics. HDBSCAN showed exceptional stability—all 6 parameter combinations tested yielded identical results.

#### Four Natural Seismic Behavior Categories

HDBSCAN discovered 4 distinct earthquake behavior types:

| Cluster | Name | Events | % of Catalog | Interpretation |
|---------|------|--------|--------------|----------------|
| **C3** | Background Seismicity | 88,451 | 98.2% | Normal tectonic activity following Gutenberg-Richter distribution |
| **C2** | Elevated Activity | 493 | 0.5% | Events during periods of heightened seismic activity |
| **C1** | Aftershock-like | 150 | 0.2% | Events with temporal clustering characteristics |
| **C0** | Anomalous | 78 | 0.1% | Unusual seismic characteristics requiring individual review |

#### Feature Importance

The most discriminating features for cluster separation:

| Rank | Feature | Cluster Separation Power |
|------|---------|-------------------------|
| 1 | `local_density` | Distinguishes sequence events from isolated ones |
| 2 | `global_interevent_hrs` | Separates aftershock-like behavior (C1) |
| 3 | `events_last_7d` | Identifies elevated activity (C2) |
| 4 | `magnitude_zscore` | Flags anomalous magnitudes (C0) |
| 5 | `is_deep` | Separates deep subduction events |

#### Hyperparameter Tuning

Automated grid search optimized parameters:

- **K-Means**: Tested k ∈ [3, 5, 7, 9], optimal k=5
- **DBSCAN**: eps ∈ [0.34, 0.41, 0.52, 0.68, 0.90] (from k-NN analysis), optimal eps=0.90, min_samples=10
- **HDBSCAN**: min_cluster_size ∈ [15, 30, 50], min_samples ∈ [5, 10], optimal: min_cluster_size=15, min_samples=5

**Key insight**: HDBSCAN showed remarkable stability—all 6 parameter combinations produced identical results (4 clusters, score=0.853), indicating robust underlying cluster structure.

### Visualizations

The experiment generated comprehensive visualizations in `clustering_results_100k_datapoints/`:

#### Algorithm Comparison
- **`algorithm_comparison.png`**: Side-by-side comparison showing HDBSCAN produces clearest separation
- **`philippine_map_comparison.png`**: Geographic distribution of clusters across all three algorithms

#### Cluster Analysis
- **`cluster_stats_HDBSCAN_tuned.png`**: Statistical profiles showing distinct characteristics for each of the 4 clusters
- **`cluster_stats_KMEANS_tuned.png`**: K-Means cluster statistics (5 clusters)
- **`cluster_stats_DBSCAN_tuned.png`**: DBSCAN cluster statistics (9 clusters + noise)

#### Spatial Visualization
- **`philippine_map_HDBSCAN_tuned.png`**: HDBSCAN clusters overlaid on Philippine map—demonstrates clusters are not geographically confined
- **`philippine_map_KMEANS_tuned.png`**: K-Means spatial distribution
- **`philippine_map_DBSCAN_tuned.png`**: DBSCAN spatial distribution

#### Temporal Analysis
- **`temporal_profiles_HDBSCAN_tuned.png`**: Temporal characteristics showing C1 (aftershock-like) exhibits classic aftershock timing patterns
- **`temporal_profiles_KMEANS_tuned.png`**: K-Means temporal profiles
- **`temporal_profiles_DBSCAN_tuned.png`**: DBSCAN temporal profiles

#### Feature Analysis
- **`feature_distributions.png`**: Distribution histograms of all 15 optimized location-agnostic features
- **`selected_features_correlation_matrix.png`**: Correlation matrix of selected features
- **`dimensionality_reduction.png`**: PCA analysis showing 95.3% variance retained in 10 components

#### Hyperparameter Tuning
- **`grid_search_visualization.png`**: Hyperparameter search results demonstrating HDBSCAN stability across all parameters
- **`knn_distance_plot.png`**: k-NN distance analysis for data-driven DBSCAN eps selection

### Key Insights

1. **HDBSCAN significantly outperforms other algorithms**: 156% improvement in silhouette score over K-Means (0.725 vs 0.283)

2. **Four natural seismic behavior categories exist**: The clustering reveals distinct earthquake behavior types that transcend geographic boundaries

3. **15 optimized features outperform 41 features**: Feature reduction improved clustering quality by 32% (silhouette: 0.551 → 0.725)

4. **Location-agnostic clustering reveals cross-regional patterns**: Events with similar seismic characteristics cluster together regardless of geographic location

5. **Robust cluster structure**: HDBSCAN's stability across all parameter combinations indicates genuine data structure, not parameter artifacts

## 🏗️ Project Structure

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
├── clustering_results_100k_datapoints/  # Analysis results
│   ├── ANALYSIS.md             # Detailed analysis report
│   ├── CLUSTER_INTERPRETATION.md  # Cluster interpretation guide
│   ├── EVAL.md                 # Project evaluation and findings
│   ├── *.png                   # Visualization outputs
│   └── *.csv                   # Metrics and cluster data
│
└── requirements.txt            # Python dependencies
```

## 🚀 Installation

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

## 📖 Usage

### Running the Clustering Experiment

```bash
python clustering_experiment.py
```

The script will:
1. Load earthquake data from CSV
2. Engineer location-agnostic features (15 optimized features)
3. Perform dimensionality reduction (PCA: 15 → 10 components)
4. Run grid search for hyperparameter tuning (25 total combinations)
5. Evaluate clustering quality using multiple metrics
6. Generate visualizations and save results to `clustering_results_100k_datapoints/`

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

## 🔬 Methodology

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

The 15 optimized location-agnostic features:

| Category | Features | Count |
|----------|----------|-------|
| **Core Physical** | `magnitude`, `depth` | 2 |
| **Normalized/Z-scores** | `magnitude_zscore`, `depth_zscore` | 2 |
| **Depth Classification** | `is_deep` | 1 |
| **Magnitude Significance** | `is_significant`, `is_major` | 2 |
| **Magnitude Dynamics** | `mag_deviation_from_recent`, `mag_diff_from_prev`, `mag_rolling_std_50` | 3 |
| **Spatial Activity** | `local_density`, `events_last_7d`, `events_last_30d` | 3 |
| **Temporal Context** | `hrs_since_significant_event`, `global_interevent_hrs` | 2 |
| **Total** | | **15** |

### Hyperparameter Tuning

Automated grid search optimizes:
- **K-Means**: Number of clusters (k), initialization runs
- **DBSCAN**: Epsilon (eps) from k-NN analysis, minimum samples
- **HDBSCAN**: Minimum cluster size, minimum samples, cluster selection method

Optimization metric: `combined_with_penalty` (balances silhouette score, Calinski-Harabasz, Davies-Bouldin, and noise penalty)

### Validation Strategy

Since unsupervised learning has no ground truth labels, we used **internal validation metrics**:
- **Silhouette Score**: Cluster cohesion vs. separation (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Davies-Bouldin Index**: Average similarity of each cluster to most similar cluster (lower is better)

## 📈 Key Insights

1. **HDBSCAN is the recommended algorithm**: Achieves highest silhouette score (0.725) with minimal noise (0%), finding a clear 4-cluster structure

2. **Four natural seismic behavior categories**: Background seismicity (98.2%), Elevated activity (0.5%), Aftershock-like (0.2%), and Anomalous (0.1%)

3. **Feature optimization matters**: Reducing from 41 to 15 features improved clustering quality by 32%

4. **Robust cluster structure**: HDBSCAN's stability across all parameter combinations indicates genuine data structure

5. **Cross-regional patterns**: Location-agnostic clustering reveals behavioral patterns that spatial clustering misses

## 📝 Documentation

Detailed analysis documents are available in `clustering_results_100k_datapoints/`:

- **`ANALYSIS.md`**: Comprehensive technical analysis including methodology, hyperparameter choices, and results
- **`CLUSTER_INTERPRETATION.md`**: Practical guide for interpreting clusters, with applications for seismologists, LGUs, and stakeholders
- **`EVAL.md`**: Project evaluation, limitations, potential applications, and future work recommendations

## 📊 Results Summary

### Computational Resources
- **System**: 8-core CPU, 16GB RAM
- **Peak Memory**: 1.16 GB
- **Total Runtime**: 16.4 minutes
- **Grid Search Time**: 13.1 minutes (79.8% of total)

### Model Performance
- **Best Algorithm**: HDBSCAN (Combined Score: 0.853)
- **Clusters Discovered**: 4 distinct seismic behavior types
- **Cluster Quality**: Excellent (Silhouette: 0.725)
- **Parameter Stability**: Exceptional (identical results across all tested parameters)

## 📝 References

- **Data Source**: [PHIVOLCS](https://earthquake.phivolcs.dost.gov.ph/) - Philippine Institute of Volcanology and Seismology
- **Clustering Algorithms**: HDBSCAN, DBSCAN, K-Means via scikit-learn
- **Analysis Results**: See `clustering_results_100k_datapoints/ANALYSIS.md`, `CLUSTER_INTERPRETATION.md`, and `EVAL.md` for detailed findings

## 👤 Author

Maxell Milay

## 📄 License

[Add your license here]

---

For detailed analysis results, see:
- `clustering_results_100k_datapoints/ANALYSIS.md` - Comprehensive technical analysis
- `clustering_results_100k_datapoints/CLUSTER_INTERPRETATION.md` - Practical cluster interpretation guide
- `clustering_results_100k_datapoints/EVAL.md` - Project evaluation and future work
