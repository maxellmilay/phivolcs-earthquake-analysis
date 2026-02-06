"""
Location-Agnostic Seismic Clustering using Machine Learning
===========================================================

This script implements unsupervised clustering for earthquake/aftershock data
from the Philippines (PHIVOLCS catalog) using ONLY location-agnostic features.

By excluding raw latitude/longitude coordinates from clustering, we discover
patterns based on seismic characteristics (magnitude, depth, temporal patterns)
rather than simply grouping events by geographic proximity.

1. Feature Engineering (Location-Agnostic):
   - Magnitude and depth
   - Local quake density (derived neighborhood metric)
   - Interevent time features (temporal)
   - Event count in temporal windows (1, 7, 30 days)
   - Temporal features (hour, day of week, etc.)

2. Clustering Algorithms:
   - K-Means
   - DBSCAN
   - HDBSCAN

3. Evaluation Metrics:
   - Silhouette Score
   - Calinski-Harabasz Index
   - Davies-Bouldin Index

4. Visualizations:
   - Spatial cluster maps (for visualization only, not clustering)
   - Temporal profiles
   - Cluster statistics
   - Algorithm comparison

Author: Maxell Milay
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
import logging
import time
import sys
import psutil
import joblib
import os


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str = "clustering_results", log_to_console: bool = True) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Parameters
    ----------
    output_dir : str
        Directory to save log file
    log_to_console : bool
        Whether to also print to console
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{output_dir}/clustering_run_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('clustering_experiment')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Log file created: {log_filename}")
    return logger


def log_system_info(logger: logging.Logger):
    """Log system information for reproducibility."""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info(f"Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    

def log_hyperparameters(logger: logging.Logger, params: Dict):
    """Log hyperparameters used in the experiment."""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 60)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"Completed: {self.name} in {self.elapsed:.2f} seconds")
        return False

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import BallTree

# For HDBSCAN - install with: pip install hdbscan
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not installed. Install with: pip install hdbscan")

# For geospatial calculations
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


# For map visualization
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    warnings.warn("Cartopy not installed. Install with: pip install cartopy")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in km).
    
    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point (in degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (in degrees)
    
    Returns
    -------
    float
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances for a set of coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_samples, 2) with [latitude, longitude]
    
    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples) in kilometers
    """
    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            distances[i, j] = d
            distances[j, i] = d
    
    return distances


# =============================================================================
# K-NN DISTANCE ANALYSIS FOR EPS SELECTION
# =============================================================================

def compute_knn_distances(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute k-th nearest neighbor distances for each point.
    
    This is used to determine appropriate eps values for DBSCAN.
    The "elbow" in the sorted k-distance plot suggests a good eps value.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    k : int
        Number of nearest neighbors (typically same as min_samples)
    
    Returns
    -------
    np.ndarray
        Sorted k-th nearest neighbor distances
    """
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because point is its own neighbor
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Get k-th neighbor distance (exclude self at index 0)
    k_distances = distances[:, k]
    
    return np.sort(k_distances)


def suggest_eps_range(X: np.ndarray, k: int = 5, 
                      percentiles: List[float] = [10, 25, 50, 75, 90]) -> Dict:
    """
    Suggest eps range for DBSCAN based on k-distance analysis.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    k : int
        Number of nearest neighbors
    percentiles : List[float]
        Percentiles to compute for eps suggestions
    
    Returns
    -------
    Dict
        Dictionary with suggested eps values and statistics
    """
    k_distances = compute_knn_distances(X, k)
    
    suggestions = {
        'k': k,
        'min': k_distances.min(),
        'max': k_distances.max(),
        'mean': k_distances.mean(),
        'median': np.median(k_distances),
        'std': k_distances.std(),
    }
    
    for p in percentiles:
        suggestions[f'p{int(p)}'] = np.percentile(k_distances, p)
    
    # Suggest a range based on percentiles (25th to 75th is usually reasonable)
    suggestions['suggested_range'] = [
        suggestions['p25'],
        suggestions['p50'],
        suggestions['p75']
    ]
    
    return suggestions


def plot_knn_distance(X: np.ndarray, k_values: List[int] = [3, 5, 10],
                      output_dir: Optional[str] = None) -> plt.Figure:
    """
    Plot k-NN distance graph for multiple k values to help select eps.
    
    The "elbow" point in the graph suggests a good eps value for DBSCAN.
    Points before the elbow will be in clusters, points after will be noise.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    k_values : List[int]
        Different k values to plot
    output_dir : str, optional
        Directory to save the plot
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    print("\nComputing k-NN distances for eps selection...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: k-distance curves for different k values
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))
    
    for k, color in zip(k_values, colors):
        k_distances = compute_knn_distances(X, k)
        ax.plot(range(len(k_distances)), k_distances, 
                label=f'k={k}', color=color, linewidth=1.5)
    
    ax.set_xlabel('Points (sorted by distance)')
    ax.set_ylabel('k-th Nearest Neighbor Distance')
    ax.set_title('k-NN Distance Plot for Eps Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view with suggested eps regions
    ax = axes[1]
    k = k_values[len(k_values) // 2]  # Use middle k value
    k_distances = compute_knn_distances(X, k)
    suggestions = suggest_eps_range(X, k)
    
    ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1.5, label=f'k={k}')
    
    # Mark suggested eps values
    for p, color, label in [(25, 'green', 'p25'), (50, 'orange', 'p50'), (75, 'red', 'p75')]:
        eps_val = suggestions[f'p{p}']
        ax.axhline(y=eps_val, color=color, linestyle='--', alpha=0.7,
                   label=f'{label}: eps={eps_val:.3f}')
    
    ax.set_xlabel('Points (sorted by distance)')
    ax.set_ylabel(f'{k}-th Nearest Neighbor Distance')
    ax.set_title(f'Suggested Eps Values (k={k})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add text box with suggestions
    textstr = f"Suggested eps range:\n  Low: {suggestions['p25']:.3f}\n  Mid: {suggestions['p50']:.3f}\n  High: {suggestions['p75']:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right', bbox=props)
    
    plt.suptitle('DBSCAN Eps Parameter Selection via k-NN Distance Analysis', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        fig.savefig(f"{output_dir}/knn_distance_plot.png", dpi=150, bbox_inches='tight')
        print(f"  Saved k-NN distance plot to {output_dir}/knn_distance_plot.png")
    
    # Print suggestions
    print(f"\n  Suggested eps values for k={k}:")
    print(f"    Conservative (fewer clusters, more noise): {suggestions['p25']:.3f}")
    print(f"    Moderate: {suggestions['p50']:.3f}")
    print(f"    Aggressive (more clusters, less noise): {suggestions['p75']:.3f}")
    
    return fig



# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_earthquake_data(filepath: str) -> pd.DataFrame:
    """
    Load earthquake data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        Loaded and preprocessed earthquake data
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Convert numeric columns
    numeric_cols = ['latitude', 'longitude', 'depth', 'magnitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing essential data
    df = df.dropna(subset=['latitude', 'longitude', 'datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Loaded {len(df)} earthquake records")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Magnitude range: {df['magnitude'].min():.1f} to {df['magnitude'].max():.1f}")
    
    return df


def apply_magnitude_threshold(df: pd.DataFrame, min_magnitude: float = 2.0) -> pd.DataFrame:
    """
    Apply minimum magnitude threshold for catalog completeness.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data
    min_magnitude : float
        Minimum magnitude threshold
    
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    filtered = df[df['magnitude'] >= min_magnitude].copy()
    print(f"After magnitude filter (>= {min_magnitude}): {len(filtered)} records")
    return filtered


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_local_quake_density(df: pd.DataFrame, radius_km: float = 50.0) -> np.ndarray:
    """
    Compute local quake density using a BallTree (fast spatial lookup on a sphere).
    """
    print(f"Computing local quake density (radius={radius_km}km)...")

    # Convert lat/lon to radians for haversine
    coords_rad = np.radians(df[['latitude', 'longitude']].values)

    # Earth radius in km
    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    # Build BallTree using haversine metric
    tree = BallTree(coords_rad, metric='haversine')

    # Query neighbors within the given radius
    # counts includes the point itself, so subtract 1
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True) - 1

    return counts


def compute_interevent_features(df: pd.DataFrame, radius_km: float = 25.0, time_window_days: float = 30.0):
    """
    Efficiently compute interevent time and distance to nearest previous event
    within a spatial radius using a BallTree and temporal window pruning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data sorted by datetime (required)
    radius_km : float
        Spatial search radius (km)
    time_window_days : float
        Only consider previous events within this many days (to limit search)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (time_since_previous_hours, distance_to_previous_km)
    """
    print(f"Computing interevent features efficiently (radius={radius_km}km, time_window={time_window_days}d)...")

    n = len(df)
    coords = np.radians(df[['latitude', 'longitude']].values)
    times = df['datetime'].values

    time_since_prev = np.full(n, np.nan)
    dist_to_prev = np.full(n, np.nan)

    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM
    tree = BallTree(coords, metric='haversine')

    # Convert time window to timedelta for pruning
    time_window_ns = np.timedelta64(int(time_window_days * 86400), 's')

    for i in range(n):
        # Skip if this is the first event
        if i == 0:
            continue

        # Only consider events within the last X days (temporal pruning)
        mask_time = (times[:i] >= times[i] - time_window_ns)
        if not np.any(mask_time):
            continue

        prev_coords = coords[:i][mask_time]

        # Build temporary BallTree on recent subset only (fast, small)
        sub_tree = BallTree(prev_coords, metric='haversine')
        query_point = coords[i].reshape(1, -1)

        # Query all neighbors within radius
        ind = sub_tree.query_radius(query_point, r=radius_rad, return_distance=True)
        distances_rad, indices = ind[1][0], ind[0][0]

        if len(indices) == 0:
            continue

        distances_km = distances_rad * EARTH_RADIUS_KM
        recent_times = times[:i][mask_time][indices]

        # Only keep earlier events
        valid_mask = recent_times < times[i]
        if not np.any(valid_mask):
            continue

        deltas = (times[i] - recent_times[valid_mask]) / np.timedelta64(1, 'h')  # hours
        nearest_idx = np.argmin(deltas)

        time_since_prev[i] = deltas[nearest_idx]
        dist_to_prev[i] = distances_km[valid_mask][nearest_idx]

    return time_since_prev, dist_to_prev


def compute_temporal_density(
    df: pd.DataFrame,
    radius_km: float = 50.0,
    windows_days: List[int] = [1, 7, 30],
    max_window_days: float = 30.0  # search horizon (largest window)
) -> Dict[int, np.ndarray]:
    """
    Efficiently compute the number of nearby events within each temporal window
    using spatial BallTree queries + temporal pruning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data (must be sorted by datetime)
    radius_km : float
        Spatial search radius (km)
    windows_days : List[int]
        Time windows (in days)
    max_window_days : float
        Maximum temporal lookback window (days) to prune search space
    
    Returns
    -------
    Dict[int, np.ndarray]
        {window_days: np.ndarray of event counts}
    """
    print(f"Computing temporal density efficiently (radius={radius_km}km, windows={windows_days}d)...")

    n = len(df)
    coords = np.radians(df[['latitude', 'longitude']].values)
    times = df['datetime'].values.astype('datetime64[ns]')

    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    # Preallocate results
    results = {w: np.zeros(n, dtype=int) for w in windows_days}
    tree = BallTree(coords, metric='haversine')

    # Convert max window to timedelta64 for filtering
    max_window = np.timedelta64(int(max_window_days * 86400), 's')

    for i in range(n):
        t_i = times[i]
        # Temporal filter: only consider events within [t_i - max_window, t_i)
        mask_time = (times[:i] >= t_i - max_window)
        if not np.any(mask_time):
            continue

        # Build small subset for recent events only
        prev_coords = coords[:i][mask_time]
        prev_times = times[:i][mask_time]

        # Query spatial neighbors
        if len(prev_coords) == 0:
            continue
        sub_tree = BallTree(prev_coords, metric='haversine')
        query_point = coords[i].reshape(1, -1)
        ind = sub_tree.query_radius(query_point, r=radius_rad, return_distance=False)

        if len(ind[0]) == 0:
            continue

        candidate_times = prev_times[ind[0]]
        deltas_days = (t_i - candidate_times) / np.timedelta64(1, 'D')

        # Count neighbors within each temporal window
        for w in windows_days:
            results[w][i] = np.sum(deltas_days <= w)

    return results


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with datetime column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional temporal features
    """
    print("Extracting temporal features...")
    result = df.copy()
    
    result['hour'] = result['datetime'].dt.hour
    result['day_of_week'] = result['datetime'].dt.dayofweek
    result['day_of_year'] = result['datetime'].dt.dayofyear
    result['month'] = result['datetime'].dt.month
    
    # Convert datetime to numeric (days since first event)
    result['days_since_start'] = (
        result['datetime'] - result['datetime'].min()
    ).dt.total_seconds() / 86400
    
    return result


def compute_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute seismic energy-based features.
    
    The energy released by an earthquake is related to magnitude by:
    log10(E) = 1.5 * M + 4.8 (Gutenberg-Richter relation)
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with magnitude column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with energy features added
    """
    print("Computing energy-based features...")
    result = df.copy()
    
    # Seismic energy in Joules (log10 scale to avoid huge numbers)
    result['log_energy'] = 1.5 * result['magnitude'] + 4.8
    
    # Energy relative to a M2.0 reference event
    reference_energy = 1.5 * 2.0 + 4.8
    result['energy_ratio_log'] = result['log_energy'] - reference_energy
    
    return result


def compute_rolling_magnitude_features(df: pd.DataFrame, 
                                        windows: List[int] = [10, 50, 100]) -> pd.DataFrame:
    """
    Compute rolling magnitude statistics (global, not spatially constrained).
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data sorted by datetime
    windows : List[int]
        Rolling window sizes (number of events)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling magnitude features
    """
    print(f"Computing rolling magnitude features (windows: {windows})...")
    result = df.copy()
    
    for w in windows:
        # Rolling mean magnitude
        result[f'mag_rolling_mean_{w}'] = result['magnitude'].rolling(
            window=w, min_periods=1
        ).mean()
        
        # Rolling std magnitude (seismic variability)
        result[f'mag_rolling_std_{w}'] = result['magnitude'].rolling(
            window=w, min_periods=1
        ).std().fillna(0)
        
        # Rolling max magnitude
        result[f'mag_rolling_max_{w}'] = result['magnitude'].rolling(
            window=w, min_periods=1
        ).max()
    
    # Magnitude deviation from recent average (anomaly indicator)
    result['mag_deviation_from_recent'] = (
        result['magnitude'] - result['mag_rolling_mean_50']
    )
    
    return result


def compute_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute depth-related features including seismological depth classification.
    
    Standard depth classifications:
    - Shallow: 0-70 km (most damaging, crustal earthquakes)
    - Intermediate: 70-300 km (subduction zone)
    - Deep: >300 km (deep subduction, less common)
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with depth column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with depth features
    """
    print("Computing depth-based features...")
    result = df.copy()
    
    # Depth classification (one-hot encoded numerically)
    result['is_shallow'] = (result['depth'] <= 70).astype(int)
    result['is_intermediate'] = ((result['depth'] > 70) & (result['depth'] <= 300)).astype(int)
    result['is_deep'] = (result['depth'] > 300).astype(int)
    
    # Log depth (depth distribution is often log-normal)
    result['log_depth'] = np.log1p(result['depth'])
    
    # Depth z-score (how anomalous is this depth)
    depth_mean = result['depth'].mean()
    depth_std = result['depth'].std()
    result['depth_zscore'] = (result['depth'] - depth_mean) / depth_std
    
    return result


def compute_magnitude_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute magnitude anomaly/statistical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with magnitude column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with magnitude anomaly features
    """
    print("Computing magnitude anomaly features...")
    result = df.copy()
    
    # Magnitude z-score (global)
    mag_mean = result['magnitude'].mean()
    mag_std = result['magnitude'].std()
    result['magnitude_zscore'] = (result['magnitude'] - mag_mean) / mag_std
    
    # Magnitude percentile (what fraction of events are smaller)
    result['magnitude_percentile'] = result['magnitude'].rank(pct=True)
    
    # Is significant event (M >= 5.0 is commonly used threshold)
    result['is_significant'] = (result['magnitude'] >= 5.0).astype(int)
    
    # Is major event (M >= 6.0)
    result['is_major'] = (result['magnitude'] >= 6.0).astype(int)
    
    return result


def compute_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical features using sin/cos transformation.
    
    This is better for ML models than raw values because it preserves
    the cyclical nature (e.g., hour 23 is close to hour 0).
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with hour, day_of_week, month columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical encoding features
    """
    print("Computing cyclical encodings...")
    result = df.copy()
    
    # Hour of day (0-23)
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    
    # Day of week (0-6)
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    
    # Month (1-12)
    result['month_sin'] = np.sin(2 * np.pi * (result['month'] - 1) / 12)
    result['month_cos'] = np.cos(2 * np.pi * (result['month'] - 1) / 12)
    
    # Day of year (1-365)
    result['doy_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result['doy_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
    
    return result


def compute_global_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global temporal features (not spatially constrained).
    
    These capture overall seismic activity patterns regardless of location.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data sorted by datetime
    
    Returns
    -------
    pd.DataFrame
        DataFrame with global temporal features
    """
    print("Computing global temporal features...")
    result = df.copy()
    n = len(result)
    
    # Time since previous event (globally, in hours)
    time_diffs = result['datetime'].diff()
    result['global_interevent_hrs'] = time_diffs.dt.total_seconds() / 3600
    result['global_interevent_hrs'] = result['global_interevent_hrs'].fillna(
        result['global_interevent_hrs'].median()
    )
    
    # Log of interevent time (often log-normal distributed)
    result['log_interevent_hrs'] = np.log1p(result['global_interevent_hrs'])
    
    # Time since last significant event (M >= 5.0)
    significant_mask = result['magnitude'] >= 5.0
    time_since_significant = np.full(n, np.nan)
    last_significant_time = None
    
    for i in range(n):
        if last_significant_time is not None:
            time_diff = (result['datetime'].iloc[i] - last_significant_time).total_seconds() / 3600
            time_since_significant[i] = time_diff
        if significant_mask.iloc[i]:
            last_significant_time = result['datetime'].iloc[i]
    
    result['hrs_since_significant_event'] = time_since_significant
    result['hrs_since_significant_event'] = result['hrs_since_significant_event'].fillna(
        result['hrs_since_significant_event'].max()  # Fill with max for events before first significant
    )
    
    # Rolling event rate (events per day in last N events)
    # First convert timedelta to numeric (days) before rolling
    time_diffs_days = result['datetime'].diff().dt.total_seconds() / 86400
    time_diffs_days = time_diffs_days.fillna(0)
    
    for window in [20, 50, 100]:
        # Time span of last N events in days
        time_spans_days = time_diffs_days.rolling(window=window, min_periods=1).sum()
        result[f'event_rate_last_{window}'] = window / time_spans_days.replace(0, np.nan)
        result[f'event_rate_last_{window}'] = result[f'event_rate_last_{window}'].fillna(
            result[f'event_rate_last_{window}'].median()
        )
    
    return result


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that help identify mainshock-aftershock sequences.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data sorted by datetime
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sequence features
    """
    print("Computing sequence features...")
    result = df.copy()
    n = len(result)
    
    # Magnitude difference from previous event
    result['mag_diff_from_prev'] = result['magnitude'].diff().fillna(0)
    
    # Cumulative magnitude sum (proxy for cumulative energy release)
    result['cumulative_mag_sum'] = result['magnitude'].cumsum()
    
    # Running count of events
    result['event_number'] = range(1, n + 1)
    
    # Fraction of catalog elapsed (0 to 1)
    result['catalog_progress'] = result['event_number'] / n
    
    # Magnitude relative to rolling max (potential aftershock indicator)
    # If current mag is much smaller than recent max, might be aftershock
    result['mag_below_recent_max'] = (
        result['mag_rolling_max_50'] - result['magnitude']
    )
    
    return result


def engineer_features(df: pd.DataFrame, 
                      density_radius_km: float = 50.0,
                      interevent_radius_km: float = 25.0,
                      temporal_windows: List[int] = [1, 7, 30],
                      rolling_windows: List[int] = [10, 50, 100]) -> pd.DataFrame:
    """
    Apply all feature engineering steps for location-agnostic clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw earthquake data
    density_radius_km : float
        Radius for local density calculation
    interevent_radius_km : float
        Radius for interevent calculations
    temporal_windows : List[int]
        Time windows for temporal density (days)
    rolling_windows : List[int]
        Rolling windows for magnitude statistics (number of events)
    
    Returns
    -------
    pd.DataFrame
        Data with engineered features
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING (LOCATION-AGNOSTIC)")
    print("=" * 60)
    
    # 1. Basic temporal features
    result = extract_temporal_features(df)
    
    # 2. Energy-based features (from magnitude)
    result = compute_energy_features(result)
    
    # 3. Rolling magnitude statistics
    result = compute_rolling_magnitude_features(result, rolling_windows)
    
    # 4. Depth classification and features
    result = compute_depth_features(result)
    
    # 5. Magnitude anomaly features
    result = compute_magnitude_anomaly_features(result)
    
    # 6. Cyclical encoding for temporal features
    result = compute_cyclical_encoding(result)
    
    # 7. Global temporal features
    result = compute_global_temporal_features(result)
    
    # 8. Sequence features (must come after rolling features)
    result = compute_sequence_features(result)
    
    # 9. Local quake density (spatially derived but location-agnostic output)
    result['local_density'] = compute_local_quake_density(result, density_radius_km)
    
    # 10. Interevent features (spatially derived)
    time_since, dist_to = compute_interevent_features(result, interevent_radius_km)
    result['time_since_prev_event_hrs'] = time_since
    result['dist_to_prev_event_km'] = dist_to
    
    # 11. Temporal density (spatially constrained event counts)
    temporal_densities = compute_temporal_density(result, density_radius_km, temporal_windows)
    for w, counts in temporal_densities.items():
        result[f'events_last_{w}d'] = counts
    
    # Fill NaN values with median for features that might have missing values
    nan_fill_cols = ['time_since_prev_event_hrs', 'dist_to_prev_event_km', 
                     'hrs_since_significant_event']
    for col in nan_fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(result[col].median())
    
    # Summary of engineered features
    feature_cols = [c for c in result.columns if c not in 
                    ['datetime', 'location', 'datetime_str', 'latitude', 'longitude']]
    print(f"\nTotal engineered features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    return result


# =============================================================================
# CLUSTERING ALGORITHMS
# =============================================================================

def prepare_clustering_features(df: pd.DataFrame, 
                                feature_cols: List[str],
                                scaler_type: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    Prepare and scale features for clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features
    feature_cols : List[str]
        Columns to use for clustering
    scaler_type : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    
    Returns
    -------
    Tuple[np.ndarray, object]
        (scaled_features, scaler_object)
    """
    X = df[feature_cols].values
    
    # Check for and handle NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\nWarning: Found {nan_count} NaN values in features. Imputing with column medians...")
        # Impute NaN values with column medians (more robust than mean)
        col_medians = np.nanmedian(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_medians, nan_indices[1])
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def run_kmeans(X: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Run K-Means clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled features
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Cluster labels
    """
    print(f"\nRunning K-Means (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Run DBSCAN clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled features
    eps : float
        Maximum distance between samples
    min_samples : int
        Minimum samples in neighborhood
    
    Returns
    -------
    np.ndarray
        Cluster labels (-1 for noise)
    """
    print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    return labels


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 10, 
                min_samples: int = 5) -> np.ndarray:
    """
    Run HDBSCAN clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled features
    min_cluster_size : int
        Minimum cluster size
    min_samples : int
        Number of samples in neighborhood for core points
    
    Returns
    -------
    np.ndarray
        Cluster labels (-1 for noise)
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, skipping...")
        return None
    
    print(f"\nRunning HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    return labels


def run_st_dbscan(df: pd.DataFrame, 
                  spatial_eps_km: float = 50.0,
                  temporal_eps_days: float = 7.0,
                  min_samples: int = 5) -> np.ndarray:
    """
    Run ST-DBSCAN: DBSCAN with separate spatial and temporal thresholds.
    
    This is a custom implementation that uses haversine distance for spatial
    neighborhood and temporal distance for time neighborhood.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with latitude, longitude, datetime
    spatial_eps_km : float
        Spatial epsilon in kilometers
    temporal_eps_days : float
        Temporal epsilon in days
    min_samples : int
        Minimum samples for core point
    
    Returns
    -------
    np.ndarray
        Cluster labels (-1 for noise)
    """
    print(f"\nRunning ST-DBSCAN (spatial_eps={spatial_eps_km}km, temporal_eps={temporal_eps_days}days)...")
    
    n = len(df)
    coords = df[['latitude', 'longitude']].values
    times = df['datetime'].values
    
    # Initialize labels
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    
    def get_neighbors(idx):
        """Get spatiotemporal neighbors of a point."""
        neighbors = []
        for j in range(n):
            if j == idx:
                continue
            
            # Spatial distance
            spatial_dist = haversine_distance(
                coords[idx, 0], coords[idx, 1],
                coords[j, 0], coords[j, 1]
            )
            
            # Temporal distance
            temporal_dist = abs((times[idx] - times[j]) / np.timedelta64(1, 'D'))
            
            if spatial_dist <= spatial_eps_km and temporal_dist <= temporal_eps_days:
                neighbors.append(j)
        
        return neighbors
    
    for i in range(n):
        if visited[i]:
            continue
        
        visited[i] = True
        neighbors = get_neighbors(i)
        
        if len(neighbors) < min_samples:
            continue  # Noise point
        
        # Start new cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_neighbors(q)
                
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            
            if labels[q] == -1:
                labels[q] = cluster_id
            
            j += 1
        
        cluster_id += 1
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    
    return labels


# =============================================================================
# HYPERPARAMETER TUNING - GRID SEARCH
# =============================================================================

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, 
                        metric: str = 'silhouette',
                        noise_penalty_weight: float = 0.3,
                        max_acceptable_noise_ratio: float = 0.5) -> float:
    """
    Evaluate clustering quality using internal metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    metric : str
        Metric to use: 'silhouette', 'calinski_harabasz', 'davies_bouldin', 
        'combined', 'combined_with_penalty'
    noise_penalty_weight : float
        Weight for noise penalty in 'combined_with_penalty' metric (0 to 1)
    max_acceptable_noise_ratio : float
        Noise ratio above which heavy penalties are applied
    
    Returns
    -------
    float
        Score (higher is better for silhouette/calinski_harabasz, 
        lower is better for davies_bouldin, combined is normalized)
    """
    # Filter out noise points
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_clusters = len(set(labels_valid)) if len(labels_valid) > 0 else 0
    noise_ratio = (~valid_mask).sum() / len(labels)
    
    # Need at least 2 clusters and some valid points
    if n_clusters < 2 or len(X_valid) < 10:
        return -np.inf if metric != 'davies_bouldin' else np.inf
    
    if metric == 'silhouette':
        return silhouette_score(X_valid, labels_valid)
    elif metric == 'calinski_harabasz':
        return calinski_harabasz_score(X_valid, labels_valid)
    elif metric == 'davies_bouldin':
        return -davies_bouldin_score(X_valid, labels_valid)  # Negate so higher is better
    elif metric == 'combined':
        # Combined score: normalize and average multiple metrics (no noise penalty)
        try:
            sil = silhouette_score(X_valid, labels_valid)
            ch = calinski_harabasz_score(X_valid, labels_valid)
            db = davies_bouldin_score(X_valid, labels_valid)
            
            # Normalize: silhouette is already -1 to 1
            # CH: log-scale normalize, DB: invert and normalize
            sil_norm = (sil + 1) / 2  # 0 to 1
            ch_norm = np.log1p(ch) / 10  # Rough normalization
            db_norm = 1 / (1 + db)  # 0 to 1, lower DB is better
            
            return (sil_norm + ch_norm + db_norm) / 3
        except Exception:
            return -np.inf
    elif metric == 'combined_with_penalty':
        # Combined score with noise penalty - penalizes excessive noise
        try:
            sil = silhouette_score(X_valid, labels_valid)
            ch = calinski_harabasz_score(X_valid, labels_valid)
            db = davies_bouldin_score(X_valid, labels_valid)
            
            # Normalize metrics to 0-1 range
            sil_norm = (sil + 1) / 2  # -1 to 1 -> 0 to 1
            ch_norm = min(np.log1p(ch) / 10, 1.0)  # Capped at 1
            db_norm = 1 / (1 + db)  # 0 to 1, lower DB is better
            
            # Base quality score (average of normalized metrics)
            quality_score = (sil_norm + ch_norm + db_norm) / 3
            
            # Noise penalty: gradual penalty that increases sharply above threshold
            # No penalty for noise_ratio < 0.1
            # Linear penalty from 0.1 to max_acceptable_noise_ratio
            # Heavy penalty above max_acceptable_noise_ratio
            if noise_ratio <= 0.1:
                noise_penalty = 0
            elif noise_ratio <= max_acceptable_noise_ratio:
                # Linear interpolation from 0 to noise_penalty_weight
                noise_penalty = noise_penalty_weight * (noise_ratio - 0.1) / (max_acceptable_noise_ratio - 0.1)
            else:
                # Heavy exponential penalty for excessive noise
                excess = noise_ratio - max_acceptable_noise_ratio
                noise_penalty = noise_penalty_weight + (excess * 2)  # Strong penalty
            
            # Cluster count bonus/penalty
            # Penalize having too few clusters (< 3) or too many (> 15)
            if n_clusters < 3:
                cluster_penalty = 0.1 * (3 - n_clusters)
            elif n_clusters > 15:
                cluster_penalty = 0.02 * (n_clusters - 15)
            else:
                cluster_penalty = 0
            
            final_score = quality_score - noise_penalty - cluster_penalty
            
            return final_score
        except Exception:
            return -np.inf
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_detailed_clustering_score(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed clustering scores including all metrics and penalties.
    
    Useful for understanding why a particular clustering was chosen.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    
    Returns
    -------
    Dict[str, float]
        Dictionary with all score components
    """
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_clusters = len(set(labels_valid)) if len(labels_valid) > 0 else 0
    noise_count = (~valid_mask).sum()
    noise_ratio = noise_count / len(labels)
    
    result = {
        'n_samples': len(labels),
        'n_clusters': n_clusters,
        'n_noise': noise_count,
        'noise_ratio': noise_ratio,
    }
    
    if n_clusters < 2 or len(X_valid) < 10:
        result['silhouette'] = np.nan
        result['calinski_harabasz'] = np.nan
        result['davies_bouldin'] = np.nan
        result['combined_score'] = -np.inf
        result['combined_with_penalty'] = -np.inf
        return result
    
    try:
        result['silhouette'] = silhouette_score(X_valid, labels_valid)
        result['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
        result['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
        
        # Normalized scores
        result['silhouette_norm'] = (result['silhouette'] + 1) / 2
        result['ch_norm'] = min(np.log1p(result['calinski_harabasz']) / 10, 1.0)
        result['db_norm'] = 1 / (1 + result['davies_bouldin'])
        
        result['combined_score'] = evaluate_clustering(X, labels, 'combined')
        result['combined_with_penalty'] = evaluate_clustering(X, labels, 'combined_with_penalty')
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def grid_search_kmeans(X: np.ndarray, 
                       k_range: List[int] = [3, 4, 5, 6, 7, 8, 9, 10],
                       n_init_range: List[int] = [10, 20],
                       metric: str = 'silhouette',
                       random_state: int = 42,
                       verbose: bool = True) -> Dict:
    """
    Grid search for K-Means hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    k_range : List[int]
        Range of k (number of clusters) to test
    n_init_range : List[int]
        Range of n_init values to test
    metric : str
        Optimization metric
    random_state : int
        Random seed
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict
        Best parameters, best score, and all results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRID SEARCH: K-MEANS")
        print("=" * 60)
        print(f"Testing: k={k_range}, n_init={n_init_range}")
        print(f"Optimization metric: {metric}")
    
    results = []
    best_score = -np.inf
    best_params = None
    best_labels = None
    
    total_combinations = len(k_range) * len(n_init_range)
    current = 0
    
    for k in k_range:
        for n_init in n_init_range:
            current += 1
            start_time = time.time()
            
            # Run K-Means
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
            labels = kmeans.fit_predict(X)
            
            # Evaluate
            score = evaluate_clustering(X, labels, metric)
            elapsed = time.time() - start_time
            
            n_clusters = len(set(labels))
            
            result = {
                'k': k,
                'n_init': n_init,
                'score': score,
                'n_clusters': n_clusters,
                'time': elapsed
            }
            results.append(result)
            
            if verbose:
                print(f"  [{current}/{total_combinations}] k={k}, n_init={n_init}: "
                      f"score={score:.4f}, time={elapsed:.2f}s")
            
            if score > best_score:
                best_score = score
                best_params = {'n_clusters': k, 'n_init': n_init}
                best_labels = labels.copy()
    
    if verbose:
        print(f"\nBest K-Means parameters: {best_params}")
        print(f"Best score ({metric}): {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_labels': best_labels,
        'all_results': pd.DataFrame(results),
        'algorithm': 'KMeans'
    }


def grid_search_dbscan(X: np.ndarray,
                       eps_range: List[float] = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
                       min_samples_range: List[int] = [3, 5, 7, 10, 15],
                       metric: str = 'silhouette',
                       verbose: bool = True) -> Dict:
    """
    Grid search for DBSCAN hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    eps_range : List[float]
        Range of eps values to test
    min_samples_range : List[int]
        Range of min_samples values to test
    metric : str
        Optimization metric
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict
        Best parameters, best score, and all results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRID SEARCH: DBSCAN")
        print("=" * 60)
        print(f"Testing: eps={eps_range}, min_samples={min_samples_range}")
        print(f"Optimization metric: {metric}")
    
    results = []
    best_score = -np.inf
    best_params = None
    best_labels = None
    
    total_combinations = len(eps_range) * len(min_samples_range)
    current = 0
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            current += 1
            start_time = time.time()
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Evaluate
            score = evaluate_clustering(X, labels, metric)
            elapsed = time.time() - start_time
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_ratio = n_noise / len(labels)
            
            result = {
                'eps': eps,
                'min_samples': min_samples,
                'score': score,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'time': elapsed
            }
            results.append(result)
            
            if verbose:
                print(f"  [{current}/{total_combinations}] eps={eps}, min_samples={min_samples}: "
                      f"score={score:.4f}, clusters={n_clusters}, noise={noise_ratio*100:.1f}%, "
                      f"time={elapsed:.2f}s")
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels.copy()
    
    if verbose:
        print(f"\nBest DBSCAN parameters: {best_params}")
        print(f"Best score ({metric}): {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_labels': best_labels,
        'all_results': pd.DataFrame(results),
        'algorithm': 'DBSCAN'
    }


def grid_search_hdbscan(X: np.ndarray,
                        min_cluster_size_range: List[int] = [5, 10, 15, 20, 30, 50],
                        min_samples_range: List[int] = [3, 5, 10, 15],
                        cluster_selection_method: List[str] = ['eom', 'leaf'],
                        metric: str = 'silhouette',
                        verbose: bool = True) -> Dict:
    """
    Grid search for HDBSCAN hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    min_cluster_size_range : List[int]
        Range of min_cluster_size values to test
    min_samples_range : List[int]
        Range of min_samples values to test
    cluster_selection_method : List[str]
        Cluster selection methods to test ('eom' or 'leaf')
    metric : str
        Optimization metric
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict
        Best parameters, best score, and all results
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available. Skipping grid search.")
        return None
    
    if verbose:
        print("\n" + "=" * 60)
        print("GRID SEARCH: HDBSCAN")
        print("=" * 60)
        print(f"Testing: min_cluster_size={min_cluster_size_range}")
        print(f"         min_samples={min_samples_range}")
        print(f"         cluster_selection_method={cluster_selection_method}")
        print(f"Optimization metric: {metric}")
    
    results = []
    best_score = -np.inf
    best_params = None
    best_labels = None
    best_clusterer = None
    
    total_combinations = (len(min_cluster_size_range) * len(min_samples_range) * 
                          len(cluster_selection_method))
    current = 0
    
    for mcs in min_cluster_size_range:
        for ms in min_samples_range:
            for csm in cluster_selection_method:
                current += 1
                start_time = time.time()
                
                # Run HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_method=csm,
                    prediction_data=True
                )
                labels = clusterer.fit_predict(X)
                
                # Evaluate
                score = evaluate_clustering(X, labels, metric)
                elapsed = time.time() - start_time
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                noise_ratio = n_noise / len(labels)
                
                result = {
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'cluster_selection_method': csm,
                    'score': score,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'time': elapsed
                }
                results.append(result)
                
                if verbose:
                    print(f"  [{current}/{total_combinations}] mcs={mcs}, ms={ms}, csm={csm}: "
                          f"score={score:.4f}, clusters={n_clusters}, "
                          f"noise={noise_ratio*100:.1f}%, time={elapsed:.2f}s")
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'min_cluster_size': mcs,
                        'min_samples': ms,
                        'cluster_selection_method': csm
                    }
                    best_labels = labels.copy()
                    best_clusterer = clusterer
    
    if verbose:
        print(f"\nBest HDBSCAN parameters: {best_params}")
        print(f"Best score ({metric}): {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_labels': best_labels,
        'best_clusterer': best_clusterer,
        'all_results': pd.DataFrame(results),
        'algorithm': 'HDBSCAN'
    }


def run_full_grid_search(X: np.ndarray,
                         algorithms: List[str] = ['kmeans', 'dbscan', 'hdbscan'],
                         metric: str = 'silhouette',
                         kmeans_params: Optional[Dict] = None,
                         dbscan_params: Optional[Dict] = None,
                         hdbscan_params: Optional[Dict] = None,
                         verbose: bool = True,
                         output_dir: Optional[str] = None) -> Dict:
    """
    Run grid search across multiple clustering algorithms.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    algorithms : List[str]
        Algorithms to tune: 'kmeans', 'dbscan', 'hdbscan'
    metric : str
        Optimization metric: 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'combined'
    kmeans_params : Dict, optional
        Custom parameter ranges for K-Means
    dbscan_params : Dict, optional
        Custom parameter ranges for DBSCAN
    hdbscan_params : Dict, optional
        Custom parameter ranges for HDBSCAN
    verbose : bool
        Print progress
    output_dir : str, optional
        Directory to save results
    
    Returns
    -------
    Dict
        Results for all algorithms with best parameters
    """
    if verbose:
        print("\n" + "=" * 60)
        print("AUTOMATED HYPERPARAMETER TUNING")
        print("=" * 60)
        print(f"Algorithms: {algorithms}")
        print(f"Metric: {metric}")
        print(f"Data shape: {X.shape}")
    
    all_results = {}
    start_time = time.time()
    
    # K-Means
    if 'kmeans' in algorithms:
        params = kmeans_params or {}
        result = grid_search_kmeans(
            X,
            k_range=params.get('k_range', [3, 4, 5, 6, 7, 8, 9, 10]),
            n_init_range=params.get('n_init_range', [10, 20]),
            metric=metric,
            verbose=verbose
        )
        all_results['kmeans'] = result
    
    # DBSCAN
    if 'dbscan' in algorithms:
        params = dbscan_params or {}
        result = grid_search_dbscan(
            X,
            eps_range=params.get('eps_range', [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]),
            min_samples_range=params.get('min_samples_range', [3, 5, 7, 10, 15]),
            metric=metric,
            verbose=verbose
        )
        all_results['dbscan'] = result
    
    # HDBSCAN
    if 'hdbscan' in algorithms:
        params = hdbscan_params or {}
        result = grid_search_hdbscan(
            X,
            min_cluster_size_range=params.get('min_cluster_size_range', [5, 10, 15, 20, 30, 50]),
            min_samples_range=params.get('min_samples_range', [3, 5, 10, 15]),
            cluster_selection_method=params.get('cluster_selection_method', ['eom', 'leaf']),
            metric=metric,
            verbose=verbose
        )
        all_results['hdbscan'] = result
    
    total_time = time.time() - start_time
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("GRID SEARCH SUMMARY")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds")
        print("\nBest results per algorithm:")
        
        for algo, result in all_results.items():
            if result is not None:
                print(f"\n  {algo.upper()}:")
                print(f"    Best params: {result['best_params']}")
                print(f"    Best score ({metric}): {result['best_score']:.4f}")
        
        # Find overall best
        best_algo = None
        best_overall_score = -np.inf
        for algo, result in all_results.items():
            if result is not None and result['best_score'] > best_overall_score:
                best_overall_score = result['best_score']
                best_algo = algo
        
        if best_algo:
            print(f"\n  OVERALL BEST: {best_algo.upper()} with score {best_overall_score:.4f}")
    
    # Save results if output_dir provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for algo, result in all_results.items():
            if result is not None:
                result['all_results'].to_csv(
                    f"{output_dir}/grid_search_{algo}.csv", index=False
                )
        
        # Save summary
        summary = []
        for algo, result in all_results.items():
            if result is not None:
                summary.append({
                    'algorithm': algo,
                    'best_score': result['best_score'],
                    **result['best_params']
                })
        pd.DataFrame(summary).to_csv(f"{output_dir}/grid_search_summary.csv", index=False)
    
    all_results['total_time'] = total_time
    all_results['metric'] = metric
    
    return all_results


def plot_grid_search_results(grid_results: Dict, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Visualize grid search results.
    
    Parameters
    ----------
    grid_results : Dict
        Results from run_full_grid_search
    output_dir : str, optional
        Directory to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    algorithms = [k for k in grid_results.keys() 
                  if k not in ['total_time', 'metric'] and grid_results[k] is not None]
    n_algos = len(algorithms)
    
    if n_algos == 0:
        return None
    
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
    if n_algos == 1:
        axes = [axes]
    
    for ax, algo in zip(axes, algorithms):
        result_df = grid_results[algo]['all_results']
        best_params = grid_results[algo]['best_params']
        
        if algo == 'kmeans':
            # Plot k vs score
            pivot = result_df.groupby('k')['score'].mean()
            ax.bar(pivot.index, pivot.values, color='steelblue', alpha=0.7)
            ax.axvline(x=best_params['n_clusters'], color='red', linestyle='--', 
                       label=f"Best k={best_params['n_clusters']}")
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel(f'Score ({grid_results["metric"]})')
            ax.set_title('K-Means: k vs Score')
            ax.legend()
            
        elif algo == 'dbscan':
            # Heatmap of eps vs min_samples
            pivot = result_df.pivot_table(values='score', index='min_samples', 
                                          columns='eps', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title('DBSCAN: eps vs min_samples')
            
        elif algo == 'hdbscan':
            # Heatmap of min_cluster_size vs min_samples (for best selection method)
            best_csm = best_params.get('cluster_selection_method', 'eom')
            filtered = result_df[result_df['cluster_selection_method'] == best_csm]
            pivot = filtered.pivot_table(values='score', index='min_samples',
                                         columns='min_cluster_size', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'HDBSCAN ({best_csm}): min_cluster_size vs min_samples')
    
    plt.suptitle(f'Grid Search Results (metric: {grid_results["metric"]})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        fig.savefig(f"{output_dir}/grid_search_visualization.png", dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute internal clustering evaluation metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Filter out noise points for metric calculation
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    n_clusters = len(set(labels_valid))
    
    if n_clusters < 2 or len(X_valid) < 2:
        print("  Warning: Not enough clusters or samples for metrics")
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': int(np.sum(~valid_mask))
        }
    
    # Silhouette Score: Higher is better (-1 to 1)
    metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
    
    # Calinski-Harabasz Index: Higher is better
    metrics['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
    
    # Davies-Bouldin Index: Lower is better
    metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = int(np.sum(~valid_mask))
    
    return metrics


def print_metrics_comparison(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Print comparison of clustering metrics across algorithms.
    
    Parameters
    ----------
    all_metrics : Dict[str, Dict[str, float]]
        Dictionary of algorithm name to metrics
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    print("\n" + "=" * 60)
    print("CLUSTERING METRICS COMPARISON")
    print("=" * 60)
    
    df = pd.DataFrame(all_metrics).T
    df = df.round(4)
    
    print("\nMetric Descriptions:")
    print("  - Silhouette Score: -1 to 1, higher is better")
    print("  - Calinski-Harabasz: Higher is better (ratio of between/within cluster dispersion)")
    print("  - Davies-Bouldin: Lower is better (average similarity of each cluster with its most similar)")
    print()
    print(df.to_string())
    
    return df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_spatial_clusters(df: pd.DataFrame, labels: np.ndarray, 
                          title: str, ax: Optional[plt.Axes] = None,
                          show_noise: bool = True) -> plt.Axes:
    """
    Plot earthquake clusters on a spatial map.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes
    show_noise : bool
        Whether to show noise points
    
    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            if show_noise:
                col = 'lightgray'
                alpha = 0.3
                marker = 'x'
                size = 20
            else:
                continue
        else:
            alpha = 0.7
            marker = 'o'
            size = df['magnitude'].iloc[labels == k] ** 2 * 10
        
        mask = labels == k
        ax.scatter(
            df['longitude'].iloc[mask],
            df['latitude'].iloc[mask],
            c=[col],
            s=size if k != -1 else 20,
            alpha=alpha,
            marker=marker,
            label=f'Cluster {k}' if k != -1 else 'Noise',
            edgecolors='k' if k != -1 else 'none',
            linewidths=0.3
        )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    return ax


def plot_clusters_on_philippine_map(df: pd.DataFrame, labels: np.ndarray,
                                     title: str = "Earthquake Clusters - Philippines",
                                     extent: List[float] = [116, 128, 4, 22],
                                     figsize: Tuple[int, int] = (12, 14),
                                     show_noise: bool = True) -> Optional[plt.Figure]:
    """
    Plot earthquake clusters on a geographic map of the Philippines using Cartopy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with latitude and longitude columns
    labels : np.ndarray
        Cluster labels for each earthquake
    title : str
        Plot title
    extent : List[float]
        Map extent as [lon_min, lon_max, lat_min, lat_max]
        Default covers the Philippines region
    figsize : Tuple[int, int]
        Figure size (width, height) in inches
    show_noise : bool
        Whether to show noise points (cluster = -1)
    
    Returns
    -------
    plt.Figure or None
        Matplotlib figure, or None if cartopy is not available
    """
    if not CARTOPY_AVAILABLE:
        print("Cartopy not available. Skipping Philippine map visualization.")
        print("Install with: pip install cartopy")
        return None
    
    print(f"Plotting clusters on Philippine map...")
    
    # Create figure with Cartopy projection
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set map extent to Philippines region
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='#666666')
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Get unique clusters and assign colors
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    color_idx = 0
    for k in unique_labels:
        mask = labels == k
        
        if k == -1:
            if show_noise:
                ax.scatter(
                    df.loc[mask, 'longitude'], 
                    df.loc[mask, 'latitude'],
                    c='lightgray',
                    s=15,
                    alpha=0.4,
                    marker='x',
                    linewidths=0.5,
                    label=f'Noise ({mask.sum()})',
                    transform=ccrs.PlateCarree(),
                    zorder=1
                )
        else:
            # Size based on magnitude
            sizes = df.loc[mask, 'magnitude'] ** 2 * 8
            ax.scatter(
                df.loc[mask, 'longitude'], 
                df.loc[mask, 'latitude'],
                c=[colors[color_idx]],
                s=sizes,
                alpha=0.65,
                edgecolors='black',
                linewidths=0.3,
                label=f'Cluster {k} ({mask.sum()})',
                transform=ccrs.PlateCarree(),
                zorder=2
            )
            color_idx += 1
    
    # Add legend
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9, ncol=2)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    return fig


def plot_clusters_on_philippine_map_comparison(df: pd.DataFrame,
                                                all_labels: Dict[str, np.ndarray],
                                                extent: List[float] = [116, 128, 4, 22]) -> Optional[plt.Figure]:
    """
    Plot comparison of different clustering algorithms on Philippine maps.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data
    all_labels : Dict[str, np.ndarray]
        Dictionary of algorithm name to cluster labels
    extent : List[float]
        Map extent as [lon_min, lon_max, lat_min, lat_max]
    
    Returns
    -------
    plt.Figure or None
        Matplotlib figure, or None if cartopy is not available
    """
    if not CARTOPY_AVAILABLE:
        print("Cartopy not available. Skipping Philippine map comparison.")
        return None
    
    n_algorithms = len(all_labels)
    if n_algorithms == 0:
        return None
    
    n_cols = min(2, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 10 * n_rows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    if n_algorithms == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, labels) in enumerate(all_labels.items()):
        if labels is None:
            continue
        
        ax = axes[idx]
        
        # Set map extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4, edgecolor='#666666')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', 
                          alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        
        # Get unique clusters and colors
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
        
        color_idx = 0
        for k in unique_labels:
            mask = labels == k
            
            if k == -1:
                ax.scatter(
                    df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                    c='lightgray', s=10, alpha=0.3, marker='x',
                    transform=ccrs.PlateCarree(), zorder=1
                )
            else:
                sizes = df.loc[mask, 'magnitude'] ** 2 * 5
                ax.scatter(
                    df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                    c=[colors[color_idx]], s=sizes, alpha=0.6,
                    edgecolors='k', linewidths=0.2,
                    transform=ccrs.PlateCarree(), zorder=2
                )
                color_idx += 1
        
        # Count clusters and noise
        n_noise = (labels == -1).sum()
        ax.set_title(f'{name}\n({n_clusters} clusters, {n_noise} noise)', fontsize=11, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_algorithms, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Clustering Algorithm Comparison - Philippines', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_temporal_profiles(df: pd.DataFrame, labels: np.ndarray, 
                           title: str = "Temporal Profiles by Cluster") -> plt.Figure:
    """
    Plot temporal profiles showing when each cluster's events occurred.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with datetime
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Add labels to dataframe
    plot_df = df.copy()
    plot_df['cluster'] = labels
    plot_df = plot_df[plot_df['cluster'] != -1]  # Exclude noise
    
    # 1. Events over time by cluster
    ax = axes[0, 0]
    for cluster in sorted(plot_df['cluster'].unique()):
        cluster_data = plot_df[plot_df['cluster'] == cluster]
        ax.scatter(
            cluster_data['datetime'],
            [cluster] * len(cluster_data),
            alpha=0.5,
            s=cluster_data['magnitude'] ** 2 * 5,
            label=f'Cluster {cluster}'
        )
    ax.set_xlabel('Date')
    ax.set_ylabel('Cluster')
    ax.set_title('Events Over Time by Cluster')
    ax.legend(fontsize=8)
    
    # 2. Cumulative events per cluster
    ax = axes[0, 1]
    for cluster in sorted(plot_df['cluster'].unique()):
        cluster_data = plot_df[plot_df['cluster'] == cluster].sort_values('datetime')
        cumsum = range(1, len(cluster_data) + 1)
        ax.plot(cluster_data['datetime'], cumsum, label=f'Cluster {cluster}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Events')
    ax.set_title('Cumulative Event Count by Cluster')
    ax.legend(fontsize=8)
    
    # 3. Interevent time distribution by cluster
    ax = axes[1, 0]
    if 'time_since_prev_event_hrs' in plot_df.columns:
        cluster_interevent = []
        for cluster in sorted(plot_df['cluster'].unique()):
            cluster_data = plot_df[plot_df['cluster'] == cluster]
            cluster_interevent.append(cluster_data['time_since_prev_event_hrs'].dropna())
        ax.boxplot(cluster_interevent, labels=[f'C{c}' for c in sorted(plot_df['cluster'].unique())])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Interevent Time (hours)')
        ax.set_title('Interevent Time Distribution by Cluster')
    
    # 4. Magnitude distribution by cluster
    ax = axes[1, 1]
    cluster_mags = []
    for cluster in sorted(plot_df['cluster'].unique()):
        cluster_data = plot_df[plot_df['cluster'] == cluster]
        cluster_mags.append(cluster_data['magnitude'].dropna())
    ax.boxplot(cluster_mags, labels=[f'C{c}' for c in sorted(plot_df['cluster'].unique())])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Magnitude')
    ax.set_title('Magnitude Distribution by Cluster')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_cluster_statistics(df: pd.DataFrame, labels: np.ndarray,
                            title: str = "Cluster Statistics") -> plt.Figure:
    """
    Plot detailed statistics for each cluster.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plot_df = df.copy()
    plot_df['cluster'] = labels
    
    # 1. Cluster sizes
    ax = axes[0, 0]
    cluster_counts = plot_df['cluster'].value_counts().sort_index()
    colors = ['lightgray' if idx == -1 else plt.cm.tab10(idx % 10) 
              for idx in cluster_counts.index]
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Events')
    ax.set_title('Events per Cluster')
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)
    
    # 2. Mean magnitude by cluster
    ax = axes[0, 1]
    mean_mags = plot_df.groupby('cluster')['magnitude'].mean().sort_index()
    colors = ['lightgray' if idx == -1 else plt.cm.tab10(idx % 10) 
              for idx in mean_mags.index]
    ax.bar(mean_mags.index.astype(str), mean_mags.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Mean Magnitude by Cluster')
    ax.axhline(y=plot_df['magnitude'].mean(), color='red', linestyle='--', 
               label=f'Overall Mean: {plot_df["magnitude"].mean():.2f}')
    ax.legend()
    
    # 3. Depth distribution by cluster
    ax = axes[1, 0]
    valid_clusters = [c for c in sorted(plot_df['cluster'].unique()) if c != -1]
    if valid_clusters:
        depth_data = [plot_df[plot_df['cluster'] == c]['depth'].dropna() for c in valid_clusters]
        bp = ax.boxplot(depth_data, labels=[str(c) for c in valid_clusters], patch_artist=True)
        for patch, c in zip(bp['boxes'], valid_clusters):
            patch.set_facecolor(plt.cm.tab10(c % 10))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Depth (km)')
    ax.set_title('Depth Distribution by Cluster')
    
    # 4. Cluster duration (time span)
    ax = axes[1, 1]
    valid_df = plot_df[plot_df['cluster'] != -1]
    durations = valid_df.groupby('cluster')['datetime'].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 86400  # days
    ).sort_index()
    colors = [plt.cm.tab10(idx % 10) for idx in durations.index]
    ax.bar(durations.index.astype(str), durations.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Duration (days)')
    ax.set_title('Cluster Duration (First to Last Event)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_algorithm_comparison(df: pd.DataFrame, 
                              all_labels: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot comparison of different clustering algorithms side by side.
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data
    all_labels : Dict[str, np.ndarray]
        Dictionary of algorithm name to cluster labels
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_algorithms = len(all_labels)
    n_cols = min(2, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_algorithms == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, labels) in enumerate(all_labels.items()):
        if labels is not None:
            plot_spatial_clusters(df, labels, name, axes[idx])
    
    # Hide unused axes
    for idx in range(n_algorithms, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Plot distributions of location-agnostic engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with engineered features
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # 15-feature set used for clustering (location-agnostic)
    feature_cols = [
        'magnitude', 'depth',
        'magnitude_zscore', 'depth_zscore',
        'is_deep',
        'is_significant', 'is_major',
        'mag_deviation_from_recent', 'mag_diff_from_prev', 'mag_rolling_std_50',
        'local_density', 'events_last_7d', 'events_last_30d',
        'hrs_since_significant_event', 'global_interevent_hrs',
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    
    n_cols = 4
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col}\n(mean={data.mean():.2f}, std={data.std():.2f})')
    
    # Hide unused axes
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Location-Agnostic Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_elbow_and_silhouette(X: np.ndarray, k_range: range = range(2, 11)) -> plt.Figure:
    """
    Plot elbow curve and silhouette scores for K-Means.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled features
    k_range : range
        Range of k values to test
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    print("\nComputing elbow curve and silhouette scores...")
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Elbow curve
    ax = axes[0]
    ax.plot(k_range, inertias, 'bo-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    
    # Silhouette scores
    ax = axes[1]
    ax.plot(k_range, silhouettes, 'ro-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs k')
    best_k = list(k_range)[np.argmax(silhouettes)]
    ax.axvline(x=best_k, color='green', linestyle='--', 
               label=f'Best k={best_k}')
    ax.legend()
    
    plt.tight_layout()
    
    return fig



# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_clustering_analysis(filepath: str,
                            min_magnitude: float = 2.0,
                            density_radius_km: float = 50.0,
                            interevent_radius_km: float = 25.0,
                            temporal_windows: List[int] = [1, 7, 30],
                            output_dir: str = "clustering_results",
                            max_samples: Optional[int] = None,
                            min_cluster_size: int = 15,
                            min_samples: int = 5,
                            kmeans_n_clusters: int = 5,
                            dbscan_eps: float = 0.8,
                            dbscan_min_samples: int = 5,
                            use_grid_search: bool = False,
                            grid_search_algorithms: List[str] = ['hdbscan'],
                            grid_search_metric: str = 'combined_with_penalty',
                            grid_search_params: Optional[Dict] = None,
                            # k-NN analysis for eps selection
                            compute_knn_eps_suggestion: bool = True,
                            knn_k_values: List[int] = [3, 5, 10]) -> Dict:
    """
    Run the complete clustering analysis pipeline.
    
    Parameters
    ----------
    filepath : str
        Path to earthquake data CSV
    min_magnitude : float
        Minimum magnitude threshold
    density_radius_km : float
        Radius for local density calculation
    interevent_radius_km : float
        Radius for interevent calculations
    temporal_windows : List[int]
        Time windows for temporal density
    output_dir : str
        Directory to save output figures
    max_samples : int, optional
        Maximum number of data points to use. If None, use all data.
        Useful for faster experimentation with large datasets.
    min_cluster_size : int
        HDBSCAN minimum cluster size parameter (used if grid_search disabled)
    min_samples : int
        HDBSCAN/DBSCAN minimum samples parameter (used if grid_search disabled)
    kmeans_n_clusters : int
        Number of clusters for K-Means (used if grid_search disabled)
    dbscan_eps : float
        DBSCAN epsilon (max distance) parameter (used if grid_search disabled)
    dbscan_min_samples : int
        DBSCAN minimum samples parameter (used if grid_search disabled)
    use_grid_search : bool
        If True, perform automated grid search for hyperparameter tuning.
    grid_search_algorithms : List[str]
        Algorithms to tune: 'kmeans', 'dbscan', 'hdbscan'
    grid_search_metric : str
        Metric to optimize: 'silhouette', 'calinski_harabasz', 'davies_bouldin', 
        'combined', 'combined_with_penalty' (recommended - includes noise penalty)
    grid_search_params : Dict, optional
        Custom parameter ranges for grid search, with keys like:
        - 'kmeans': {'k_range': [...], 'n_init_range': [...]}
        - 'dbscan': {'eps_range': [...], 'min_samples_range': [...]}
        - 'hdbscan': {'min_cluster_size_range': [...], 'min_samples_range': [...], ...}
    compute_knn_eps_suggestion : bool
        If True, compute and plot k-NN distances to suggest eps values for DBSCAN
    knn_k_values : List[int]
        k values to use for k-NN distance analysis
    
    Returns
    -------
    Dict
        Results including data, labels, metrics, and optionally grid_search_results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    total_start_time = time.time()
    
    # Log system info
    log_system_info(logger)
    
    # Log all hyperparameters
    hyperparams = {
        'filepath': filepath,
        'min_magnitude': min_magnitude,
        'max_samples': max_samples if max_samples else 'all',
        'density_radius_km': density_radius_km,
        'interevent_radius_km': interevent_radius_km,
        'temporal_windows': temporal_windows,
        'output_dir': output_dir,
        'hdbscan_min_cluster_size': min_cluster_size,
        'hdbscan_min_samples': min_samples,
        'kmeans_n_clusters': kmeans_n_clusters,
        'dbscan_eps': dbscan_eps,
        'dbscan_min_samples': dbscan_min_samples,
        'feature_scaler': 'StandardScaler',
        'rolling_windows': [10, 50, 100],
        'use_grid_search': use_grid_search,
        'grid_search_algorithms': grid_search_algorithms if use_grid_search else 'N/A',
        'grid_search_metric': grid_search_metric if use_grid_search else 'N/A',
        'compute_knn_eps_suggestion': compute_knn_eps_suggestion,
    }
    log_hyperparameters(logger, hyperparams)
    
    grid_search_results = None
    
    timings = {}
    
    # ==========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    with TimingContext("Data Loading", logger) as tc:
        df = load_earthquake_data(filepath)
        df = apply_magnitude_threshold(df, min_magnitude)
        
        # Limit to max_samples if specified
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples).reset_index(drop=True)
            logger.info(f"Limited to first {max_samples} records (chronologically)")
    timings['data_loading'] = tc.elapsed
    
    logger.info(f"Dataset size: {len(df)} records")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # ==========================================================================
    # 2. FEATURE ENGINEERING
    # ==========================================================================
    with TimingContext("Feature Engineering", logger) as tc:
        df = engineer_features(
            df,
            density_radius_km=density_radius_km,
            interevent_radius_km=interevent_radius_km,
            temporal_windows=temporal_windows
        )
    timings['feature_engineering'] = tc.elapsed
    
    # Plot feature distributions
    fig = plot_feature_distributions(df)
    fig.savefig(f"{output_dir}/feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ==========================================================================
    # 3. PREPARE CLUSTERING FEATURES (LOCATION-AGNOSTIC)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PREPARING CLUSTERING FEATURES (LOCATION-AGNOSTIC)")
    print("=" * 60)
    
    # Define location-agnostic feature set
    # NOTE: We intentionally EXCLUDE latitude and longitude from clustering
    # to discover patterns based on seismic characteristics, not geographic proximity.
    # This allows the model to find clusters of events with similar magnitude, depth,
    # and temporal patterns regardless of where they occurred.
    #
    # OPTIMIZED FEATURE SET (15 features)
    # Selected based on feature importance analysis and redundancy removal.
    # Clustering algorithms work best with 10-20 features due to curse of dimensionality.
    # Removed: redundant features (log_energy ~ magnitude), cyclical time encodings,
    # event rates, and features with near-zero cluster separation power.
    location_agnostic_features = [
        'magnitude',                    # Core physical: event size
        'depth',                        # Core physical: event depth
        'magnitude_zscore',             # Magnitude anomaly (global z-score)
        'depth_zscore',                 # Depth anomaly (global z-score)
        'is_deep',                      # Depth category: deep events (>300km)
        'is_significant',               # Magnitude category: M >= 5.0
        'is_major',                     # Magnitude category: M >= 6.0
        'mag_deviation_from_recent',    # Magnitude dynamics: deviation from rolling mean
        'mag_diff_from_prev',           # Magnitude dynamics: change from previous event
        'mag_rolling_std_50',           # Magnitude dynamics: recent volatility
        'local_density',                # Spatial: total earthquakes within radius
        'events_last_7d',               # Spatial-temporal: recent nearby events (7 days)
        'events_last_30d',              # Spatial-temporal: recent nearby events (30 days)
        'hrs_since_significant_event',  # Temporal: hours since last M >= 5.0
        'global_interevent_hrs',        # Temporal: hours since previous event
    ]
    
    # Use only available location-agnostic features
    feature_cols = [c for c in location_agnostic_features if c in df.columns]
    print(f"Using {len(feature_cols)} location-agnostic features:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1}. {col}")
    print("\nNOTE: latitude/longitude excluded to avoid spatial clustering bias")
    
    X_scaled, scaler = prepare_clustering_features(df, feature_cols, 'standard')
    
    # ==========================================================================
    # 4. K-NN DISTANCE ANALYSIS FOR EPS SELECTION
    # ==========================================================================
    eps_suggestions = None
    if compute_knn_eps_suggestion:
        logger.info("\n" + "=" * 60)
        logger.info("K-NN DISTANCE ANALYSIS FOR EPS SELECTION")
        logger.info("=" * 60)
        
        with TimingContext("k-NN Distance Analysis", logger) as tc:
            # Compute k-NN distances and plot
            fig = plot_knn_distance(X_scaled, k_values=knn_k_values, output_dir=output_dir)
            plt.close(fig)
            
            # Get suggested eps values for use in grid search
            k_for_suggestion = knn_k_values[len(knn_k_values) // 2]  # Use middle k value
            eps_suggestions = suggest_eps_range(X_scaled, k=k_for_suggestion)
            
            logger.info(f"  k={k_for_suggestion} eps suggestions:")
            logger.info(f"    Conservative (p25): {eps_suggestions['p25']:.3f}")
            logger.info(f"    Moderate (p50): {eps_suggestions['p50']:.3f}")
            logger.info(f"    Aggressive (p75): {eps_suggestions['p75']:.3f}")
        timings['knn_analysis'] = tc.elapsed
        
        # Update hyperparams with eps suggestions
        hyperparams['eps_suggestion_conservative'] = eps_suggestions['p25']
        hyperparams['eps_suggestion_moderate'] = eps_suggestions['p50']
        hyperparams['eps_suggestion_aggressive'] = eps_suggestions['p75']
        
        # If grid_search_params for dbscan doesn't have eps_range specified,
        # use the k-NN suggested range
        if use_grid_search and 'dbscan' in grid_search_algorithms:
            custom_params = grid_search_params or {}
            if 'dbscan' not in custom_params or 'eps_range' not in custom_params.get('dbscan', {}):
                suggested_eps_range = [
                    round(eps_suggestions['p10'], 2),
                    round(eps_suggestions['p25'], 2),
                    round(eps_suggestions['p50'], 2),
                    round(eps_suggestions['p75'], 2),
                    round(eps_suggestions['p90'], 2),
                ]
                logger.info(f"  Using k-NN suggested eps_range for DBSCAN: {suggested_eps_range}")
                if grid_search_params is None:
                    grid_search_params = {}
                if 'dbscan' not in grid_search_params:
                    grid_search_params['dbscan'] = {}
                grid_search_params['dbscan']['eps_range'] = suggested_eps_range
    
    # ==========================================================================
    # 6. RUN CLUSTERING ALGORITHMS (with optional grid search)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING CLUSTERING ALGORITHMS")
    logger.info("=" * 60)
    
    all_labels = {}
    all_metrics = {}
    
    if use_grid_search:
        # ==========================================================================
        # AUTOMATED GRID SEARCH FOR HYPERPARAMETER TUNING
        # ==========================================================================
        logger.info("\nGrid search enabled - finding optimal hyperparameters...")
        logger.info(f"Algorithms to tune: {grid_search_algorithms}")
        logger.info(f"Optimization metric: {grid_search_metric}")
        
        # Prepare custom parameter ranges
        custom_params = grid_search_params or {}
        kmeans_params = custom_params.get('kmeans', None)
        dbscan_params = custom_params.get('dbscan', None)
        hdbscan_params = custom_params.get('hdbscan', None)
        
        with TimingContext("Grid Search", logger) as tc:
            grid_search_results = run_full_grid_search(
                X_scaled,
                algorithms=grid_search_algorithms,
                metric=grid_search_metric,
                kmeans_params=kmeans_params,
                dbscan_params=dbscan_params,
                hdbscan_params=hdbscan_params,
                verbose=True,
                output_dir=output_dir
            )
        timings['grid_search'] = tc.elapsed
        
        # Plot grid search results
        fig = plot_grid_search_results(grid_search_results, output_dir)
        if fig is not None:
            plt.close(fig)
        
        # Extract best results for each algorithm
        for algo in grid_search_algorithms:
            if algo in grid_search_results and grid_search_results[algo] is not None:
                result = grid_search_results[algo]
                algo_name = f"{algo.upper()} (tuned)"
                all_labels[algo_name] = result['best_labels']
                all_metrics[algo_name] = compute_clustering_metrics(X_scaled, result['best_labels'])
                
                # Update hyperparams with best found values
                hyperparams[f'{algo}_best_params'] = result['best_params']
                hyperparams[f'{algo}_best_score'] = result['best_score']
                
                logger.info(f"\n{algo.upper()} Best Parameters: {result['best_params']}")
                logger.info(f"{algo.upper()} Best Score ({grid_search_metric}): {result['best_score']:.4f}")
        
        timings['clustering'] = tc.elapsed
        
    else:
        # ==========================================================================
        # USE PROVIDED HYPERPARAMETERS DIRECTLY
        # ==========================================================================
        
        # K-Means (commented out by default)
        # with TimingContext("K-Means Clustering", logger) as tc:
        #     labels_kmeans = run_kmeans(X_scaled, n_clusters=kmeans_n_clusters)
        #     all_labels['K-Means (k=5)'] = labels_kmeans
        #     all_metrics['K-Means (k=5)'] = compute_clustering_metrics(X_scaled, labels_kmeans)
        # timings['kmeans_clustering'] = tc.elapsed
        
        # DBSCAN (commented out by default)
        # with TimingContext("DBSCAN Clustering", logger) as tc:
        #     labels_dbscan = run_dbscan(X_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples)
        #     all_labels['DBSCAN'] = labels_dbscan
        #     all_metrics['DBSCAN'] = compute_clustering_metrics(X_scaled, labels_dbscan)
        # timings['dbscan_clustering'] = tc.elapsed
        
        # HDBSCAN
        with TimingContext("HDBSCAN Clustering", logger) as tc:
            labels_hdbscan = run_hdbscan(X_scaled, min_cluster_size=min_cluster_size, min_samples=min_samples)
            if labels_hdbscan is not None:
                all_labels['HDBSCAN'] = labels_hdbscan
                all_metrics['HDBSCAN'] = compute_clustering_metrics(X_scaled, labels_hdbscan)
        timings['hdbscan_clustering'] = tc.elapsed
    
    # Log clustering results
    for algo_name, metrics in all_metrics.items():
        logger.info(f"\n{algo_name} Results:")
        logger.info(f"  Clusters found: {metrics.get('n_clusters', 'N/A')}")
        logger.info(f"  Noise points: {metrics.get('n_noise', 'N/A')}")
        logger.info(f"  Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}" if not np.isnan(metrics.get('silhouette', np.nan)) else f"  Silhouette Score: N/A")
        logger.info(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.4f}" if not np.isnan(metrics.get('calinski_harabasz', np.nan)) else f"  Calinski-Harabasz: N/A")
        logger.info(f"  Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.4f}" if not np.isnan(metrics.get('davies_bouldin', np.nan)) else f"  Davies-Bouldin: N/A")
    
    # NOTE: ST-DBSCAN is excluded because it is inherently spatial (uses lat/lng
    # with spatial epsilon in km). For location-agnostic clustering, we rely on
    # the algorithms above that use our location-agnostic feature set.
    
    # ==========================================================================
    # 7. EVALUATE AND COMPARE
    # ==========================================================================
    metrics_df = print_metrics_comparison(all_metrics)
    metrics_df.to_csv(f"{output_dir}/clustering_metrics.csv")
    
    # ==========================================================================
    # 8. VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Algorithm comparison (simple scatter)
    fig = plot_algorithm_comparison(df, all_labels)
    fig.savefig(f"{output_dir}/algorithm_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Philippine map comparison (with geographic features)
    fig = plot_clusters_on_philippine_map_comparison(df, all_labels)
    if fig is not None:
        fig.savefig(f"{output_dir}/philippine_map_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Detailed visualizations for each algorithm
    for name, labels in all_labels.items():
        if labels is None:
            continue
        
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        
        # Philippine map for individual algorithm
        fig = plot_clusters_on_philippine_map(df, labels, f"Earthquake Clusters - {name}")
        if fig is not None:
            fig.savefig(f"{output_dir}/philippine_map_{safe_name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Temporal profiles
        fig = plot_temporal_profiles(df, labels, f"Temporal Profiles - {name}")
        fig.savefig(f"{output_dir}/temporal_profiles_{safe_name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Cluster statistics
        fig = plot_cluster_statistics(df, labels, f"Cluster Statistics - {name}")
        fig.savefig(f"{output_dir}/cluster_stats_{safe_name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ==========================================================================
    # 9. SAVE RESULTS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # Add cluster labels to dataframe
    for name, labels in all_labels.items():
        if labels is not None:
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            df[f'cluster_{safe_name}'] = labels
    
    df.to_csv(f"{output_dir}/clustered_earthquakes.csv", index=False)
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    timings['total'] = total_elapsed
    
    # Log timing summary
    logger.info("\n" + "=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    for step, elapsed in timings.items():
        logger.info(f"  {step}: {elapsed:.2f} seconds")
    
    # Log memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"\nPeak memory usage: {memory_info.rss / (1024**3):.2f} GB")
    
    # Save timings to file
    timings_df = pd.DataFrame([timings])
    timings_df.to_csv(f"{output_dir}/timing_metrics.csv", index=False)
    
    # ==========================================================================
    # 10. SAVE MODEL TO api/models
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)
    
    # Create api/models directory if it doesn't exist
    models_dir = "api/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate model filename with current datetime
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare model data to save
    model_data = {
        'timestamp': model_timestamp,
        'hyperparameters': hyperparams,
        'feature_columns': feature_cols,
        'scaler': scaler,
        'metrics': all_metrics,
        'labels': all_labels,
    }
    
    # Add best clusterer if available from grid search
    if grid_search_results is not None:
        # Save the best model from each algorithm
        for algo in grid_search_results:
            if algo not in ['total_time', 'metric'] and grid_search_results[algo] is not None:
                if 'best_clusterer' in grid_search_results[algo]:
                    model_data[f'{algo}_clusterer'] = grid_search_results[algo]['best_clusterer']
                model_data[f'{algo}_best_params'] = grid_search_results[algo]['best_params']
                model_data[f'{algo}_best_score'] = grid_search_results[algo]['best_score']
    
    # Save the model
    model_filename = f"{models_dir}/clustering_model_{model_timestamp}.joblib"
    joblib.dump(model_data, model_filename)
    logger.info(f"Model saved to: {model_filename}")
    
    # Also save a copy in the output directory
    local_model_filename = f"{output_dir}/clustering_model_{model_timestamp}.joblib"
    joblib.dump(model_data, local_model_filename)
    logger.info(f"Model copy saved to: {local_model_filename}")
    
    logger.info(f"\nResults saved to {output_dir}/")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    result = {
        'data': df,
        'labels': all_labels,
        'metrics': all_metrics,
        'metrics_df': metrics_df,
        'timings': timings,
        'hyperparameters': hyperparams,
        'feature_columns': feature_cols,
        'model_path': model_filename,
        'scaler': scaler,
    }
    
    # Add grid search results if performed
    if grid_search_results is not None:
        result['grid_search_results'] = grid_search_results
    
    # Add eps suggestions if computed
    if eps_suggestions is not None:
        result['eps_suggestions'] = eps_suggestions
    
    return result


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_FILE = "data/earthquake_data_20260203_184010.csv"
    OUTPUT_DIR = "clustering_results_100k_datapoints_backup"
    
    # Set to True to enable automated hyperparameter tuning via grid search
    USE_GRID_SEARCH = True
    
    # Run analysis
    results = run_clustering_analysis(
        filepath=DATA_FILE,
        min_magnitude=2.0,          # Magnitude threshold for completeness
        max_samples=None,           # Limit to N data points (set to None for all data)
        density_radius_km=50.0,     # Radius for local density calculation
        interevent_radius_km=25.0,  # Radius for interevent features
        temporal_windows=[1, 7, 30], # Time windows in days
        output_dir=OUTPUT_DIR,
        
        # k-NN analysis for eps selection (for DBSCAN)
        compute_knn_eps_suggestion=True,
        knn_k_values=[3, 5, 10],
        
        # Grid search options
        use_grid_search=USE_GRID_SEARCH,
        grid_search_algorithms=['kmeans', 'dbscan', 'hdbscan'],  # Algorithms to tune
        grid_search_metric='combined_with_penalty',  # Options: 'silhouette', 'calinski_harabasz', 
                                                     # 'davies_bouldin', 'combined', 'combined_with_penalty'
        # Optionally customize grid search parameter ranges:
        # Note: If compute_knn_eps_suggestion=True, eps_range for DBSCAN will be 
        # auto-populated from k-NN analysis if not specified here
        grid_search_params={
            'kmeans': {'k_range': [3, 5, 7, 9], 'n_init_range': [10]},
            # 'dbscan': {'eps_range': [...], 'min_samples_range': [5, 10]},  # eps_range auto-set by k-NN
            'dbscan': {'min_samples_range': [5, 10, 15]},  # eps_range will be set by k-NN analysis
            'hdbscan': {
                'min_cluster_size_range': [15, 30, 50],
                'min_samples_range': [5, 10],
                'cluster_selection_method': ['eom']
            }
        }
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - feature_distributions.png")
    print("  - knn_distance_plot.png (eps selection guide)")
    print("  - algorithm_comparison.png")
    print("  - philippine_map_comparison.png")
    print("  - philippine_map_*.png")
    print("  - temporal_profiles_*.png")
    print("  - cluster_stats_*.png")
    print("  - clustering_metrics.csv")
    print("  - clustered_earthquakes.csv")
    print("  - timing_metrics.csv")
    if USE_GRID_SEARCH:
        print("  - grid_search_*.csv (hyperparameter tuning results)")
        print("  - grid_search_visualization.png")
    
    # Print eps suggestions
    if 'eps_suggestions' in results:
        eps = results['eps_suggestions']
        print(f"\nDBSCAN eps suggestions (from k-NN analysis):")
        print(f"  Conservative (p25): {eps['p25']:.3f}")
        print(f"  Moderate (p50): {eps['p50']:.3f}")
        print(f"  Aggressive (p75): {eps['p75']:.3f}")
