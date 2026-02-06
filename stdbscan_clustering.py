"""
ST-DBSCAN Spatiotemporal Earthquake Clustering
===============================================

Optimized script for earthquake clustering using ST-DBSCAN (Spatiotemporal DBSCAN).
Unlike HDBSCAN which uses location-agnostic features, ST-DBSCAN explicitly uses
latitude/longitude for spatial clustering and datetime for temporal clustering.

This approach is ideal for identifying:
- Mainshock-aftershock sequences (spatially and temporally clustered)
- Earthquake swarms in specific regions
- Regional seismic activity patterns

Features Used:
- Spatial: latitude, longitude (with haversine distance)
- Temporal: datetime
- Additional: magnitude, depth (for enriched analysis)

Author: Maxell Milay
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings
from collections import deque

# Preprocessing and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import BallTree

# For map visualization
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    warnings.warn("Cartopy not installed. Map visualizations will be skipped.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on Earth (in km)."""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# =============================================================================
# DATA LOADING
# =============================================================================

def load_earthquake_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess earthquake data from CSV file."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    numeric_cols = ['latitude', 'longitude', 'depth', 'magnitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['latitude', 'longitude', 'datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Loaded {len(df)} earthquake records")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Magnitude range: {df['magnitude'].min():.1f} to {df['magnitude'].max():.1f}")
    print(f"Location range: ({df['latitude'].min():.2f}, {df['longitude'].min():.2f}) to "
          f"({df['latitude'].max():.2f}, {df['longitude'].max():.2f})")
    
    return df


def apply_magnitude_threshold(df: pd.DataFrame, min_magnitude: float = 2.0) -> pd.DataFrame:
    """Apply minimum magnitude threshold for catalog completeness."""
    filtered = df[df['magnitude'] >= min_magnitude].copy()
    print(f"After magnitude filter (>= {min_magnitude}): {len(filtered)} records")
    return filtered


# =============================================================================
# ST-DBSCAN IMPLEMENTATION (OPTIMIZED)
# =============================================================================

def run_st_dbscan_optimized(df: pd.DataFrame,
                            spatial_eps_km: float = 50.0,
                            temporal_eps_days: float = 7.0,
                            min_samples: int = 5) -> np.ndarray:
    """
    Optimized ST-DBSCAN using BallTree for spatial queries.
    
    ST-DBSCAN clusters points that are close in both space (haversine distance)
    and time (within temporal_eps_days of each other).
    
    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with latitude, longitude, datetime columns
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
    print(f"\nRunning ST-DBSCAN (spatial_eps={spatial_eps_km}km, temporal_eps={temporal_eps_days}d, min_samples={min_samples})...")
    
    n = len(df)
    coords = df[['latitude', 'longitude']].values
    coords_rad = np.radians(coords)
    
    # Convert datetime to numeric (days since first event)
    times = (df['datetime'] - df['datetime'].min()).dt.total_seconds().values / 86400
    
    # Build BallTree for spatial queries
    EARTH_RADIUS_KM = 6371.0
    spatial_eps_rad = spatial_eps_km / EARTH_RADIUS_KM
    tree = BallTree(coords_rad, metric='haversine')
    
    # Initialize
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    
    def get_spatiotemporal_neighbors(idx: int) -> List[int]:
        """Get neighbors that are within both spatial and temporal thresholds."""
        # First get spatial neighbors using BallTree (fast)
        query_point = coords_rad[idx].reshape(1, -1)
        spatial_neighbors = tree.query_radius(query_point, r=spatial_eps_rad)[0]
        
        # Then filter by temporal distance
        time_i = times[idx]
        neighbors = []
        for j in spatial_neighbors:
            if j != idx:
                time_diff = abs(times[j] - time_i)
                if time_diff <= temporal_eps_days:
                    neighbors.append(j)
        
        return neighbors
    
    # Main ST-DBSCAN loop
    for i in range(n):
        if visited[i]:
            continue
        
        visited[i] = True
        neighbors = get_spatiotemporal_neighbors(i)
        
        if len(neighbors) < min_samples:
            continue  # Noise point (for now)
        
        # Start new cluster
        labels[i] = cluster_id
        seed_set = deque(neighbors)
        
        while seed_set:
            q = seed_set.popleft()
            
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_spatiotemporal_neighbors(q)
                
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            
            if labels[q] == -1:
                labels[q] = cluster_id
        
        cluster_id += 1
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/n:.1f}%)")
    
    return labels


def run_st_dbscan_naive(df: pd.DataFrame,
                        spatial_eps_km: float = 50.0,
                        temporal_eps_days: float = 7.0,
                        min_samples: int = 5) -> np.ndarray:
    """
    Naive O(n²) ST-DBSCAN implementation for reference/small datasets.
    Uses direct haversine distance calculation.
    """
    print(f"\nRunning ST-DBSCAN naive (spatial_eps={spatial_eps_km}km, temporal_eps={temporal_eps_days}d)...")
    
    n = len(df)
    coords = df[['latitude', 'longitude']].values
    times = df['datetime'].values
    
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    
    def get_neighbors(idx):
        neighbors = []
        for j in range(n):
            if j == idx:
                continue
            
            spatial_dist = haversine_distance(
                coords[idx, 0], coords[idx, 1],
                coords[j, 0], coords[j, 1]
            )
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
            continue
        
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
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/n:.1f}%)")
    
    return labels


# =============================================================================
# FEATURE ENGINEERING (WITH LOCATION)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for ST-DBSCAN analysis.
    Includes location features since ST-DBSCAN is spatiotemporal.
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING (WITH LOCATION)")
    print("=" * 60)
    
    result = df.copy()
    n = len(result)
    
    # Temporal features
    print("Computing temporal features...")
    result['hour'] = result['datetime'].dt.hour
    result['day_of_week'] = result['datetime'].dt.dayofweek
    result['day_of_year'] = result['datetime'].dt.dayofyear
    result['month'] = result['datetime'].dt.month
    result['days_since_start'] = (result['datetime'] - result['datetime'].min()).dt.total_seconds() / 86400
    
    # Energy features
    print("Computing energy features...")
    result['log_energy'] = 1.5 * result['magnitude'] + 4.8
    
    # Depth features
    print("Computing depth features...")
    result['is_shallow'] = (result['depth'] <= 70).astype(int)
    result['is_intermediate'] = ((result['depth'] > 70) & (result['depth'] <= 300)).astype(int)
    result['is_deep'] = (result['depth'] > 300).astype(int)
    result['log_depth'] = np.log1p(result['depth'])
    
    # Global interevent time
    print("Computing interevent features...")
    time_diffs = result['datetime'].diff()
    result['global_interevent_hrs'] = time_diffs.dt.total_seconds() / 3600
    result['global_interevent_hrs'] = result['global_interevent_hrs'].fillna(
        result['global_interevent_hrs'].median()
    )
    
    # Magnitude statistics
    print("Computing magnitude statistics...")
    result['magnitude_zscore'] = (result['magnitude'] - result['magnitude'].mean()) / result['magnitude'].std()
    result['is_significant'] = (result['magnitude'] >= 5.0).astype(int)
    
    return result


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_clustering_metrics(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering metrics using spatiotemporal features.
    """
    metrics = {}
    
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))
    
    if n_clusters < 2 or np.sum(valid_mask) < 2:
        print("  Warning: Not enough clusters or samples for metrics")
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'n_clusters': n_clusters,
            'n_noise': int(np.sum(~valid_mask)),
            'noise_ratio': np.sum(~valid_mask) / len(labels)
        }
    
    # Create feature matrix for metric calculation (normalized lat/lon/time)
    X = np.column_stack([
        (df['latitude'] - df['latitude'].mean()) / df['latitude'].std(),
        (df['longitude'] - df['longitude'].mean()) / df['longitude'].std(),
        (df['days_since_start'] - df['days_since_start'].mean()) / df['days_since_start'].std()
    ])
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
    metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = int(np.sum(~valid_mask))
    metrics['noise_ratio'] = np.sum(~valid_mask) / len(labels)
    
    return metrics


def compute_cluster_characteristics(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute detailed characteristics for each cluster."""
    plot_df = df.copy()
    plot_df['cluster'] = labels
    
    stats = []
    for cluster_id in sorted(plot_df['cluster'].unique()):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        
        stat = {
            'cluster': cluster_id,
            'n_events': len(cluster_data),
            'mean_magnitude': cluster_data['magnitude'].mean(),
            'max_magnitude': cluster_data['magnitude'].max(),
            'mean_depth': cluster_data['depth'].mean(),
            'lat_center': cluster_data['latitude'].mean(),
            'lon_center': cluster_data['longitude'].mean(),
            'spatial_spread_km': 0,
            'duration_days': 0
        }
        
        if len(cluster_data) > 1:
            # Spatial spread (max distance between any two points)
            coords = cluster_data[['latitude', 'longitude']].values
            max_dist = 0
            for i in range(len(coords)):
                for j in range(i+1, min(i+50, len(coords))):  # Sample for large clusters
                    d = haversine_distance(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                    max_dist = max(max_dist, d)
            stat['spatial_spread_km'] = max_dist
            
            # Duration
            stat['duration_days'] = (cluster_data['datetime'].max() - 
                                    cluster_data['datetime'].min()).total_seconds() / 86400
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_clusters_on_map(df: pd.DataFrame, labels: np.ndarray,
                         title: str = "ST-DBSCAN Clusters - Philippines",
                         extent: List[float] = [116, 128, 4, 22],
                         figsize: Tuple[int, int] = (12, 14),
                         show_noise: bool = True) -> Optional[plt.Figure]:
    """Plot earthquake clusters on a geographic map of the Philippines."""
    if not CARTOPY_AVAILABLE:
        print("Cartopy not available. Skipping map visualization.")
        return None
    
    print("Plotting clusters on Philippine map...")
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='#666666')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    color_idx = 0
    for k in unique_labels:
        mask = labels == k
        
        if k == -1:
            if show_noise:
                ax.scatter(df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                          c='lightgray', s=15, alpha=0.4, marker='x', linewidths=0.5,
                          label=f'Noise ({mask.sum()})', transform=ccrs.PlateCarree(), zorder=1)
        else:
            sizes = df.loc[mask, 'magnitude'] ** 2 * 8
            ax.scatter(df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                      c=[colors[color_idx]], s=sizes, alpha=0.65, edgecolors='black', linewidths=0.3,
                      label=f'Cluster {k} ({mask.sum()})', transform=ccrs.PlateCarree(), zorder=2)
            color_idx += 1
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9, ncol=2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    
    return fig


def plot_spatiotemporal_view(df: pd.DataFrame, labels: np.ndarray,
                              title: str = "ST-DBSCAN Spatiotemporal View") -> plt.Figure:
    """Plot 3D-like spatiotemporal view (lat/lon over time)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_df = df.copy()
    plot_df['cluster'] = labels
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    # 1. Latitude vs Time
    ax = axes[0, 0]
    color_idx = 0
    for k in unique_labels:
        mask = labels == k
        if k == -1:
            ax.scatter(plot_df.loc[mask, 'datetime'], plot_df.loc[mask, 'latitude'],
                      c='lightgray', s=10, alpha=0.3, marker='x')
        else:
            ax.scatter(plot_df.loc[mask, 'datetime'], plot_df.loc[mask, 'latitude'],
                      c=[colors[color_idx]], s=plot_df.loc[mask, 'magnitude']**2 * 3,
                      alpha=0.6, label=f'C{k}')
            color_idx += 1
    ax.set_xlabel('Time')
    ax.set_ylabel('Latitude')
    ax.set_title('Latitude vs Time')
    ax.legend(fontsize=7, ncol=3)
    
    # 2. Longitude vs Time
    ax = axes[0, 1]
    color_idx = 0
    for k in unique_labels:
        mask = labels == k
        if k == -1:
            ax.scatter(plot_df.loc[mask, 'datetime'], plot_df.loc[mask, 'longitude'],
                      c='lightgray', s=10, alpha=0.3, marker='x')
        else:
            ax.scatter(plot_df.loc[mask, 'datetime'], plot_df.loc[mask, 'longitude'],
                      c=[colors[color_idx]], s=plot_df.loc[mask, 'magnitude']**2 * 3, alpha=0.6)
            color_idx += 1
    ax.set_xlabel('Time')
    ax.set_ylabel('Longitude')
    ax.set_title('Longitude vs Time')
    
    # 3. Magnitude over time by cluster
    ax = axes[1, 0]
    color_idx = 0
    for k in unique_labels:
        if k == -1:
            continue
        mask = labels == k
        cluster_data = plot_df[mask].sort_values('datetime')
        ax.scatter(cluster_data['datetime'], cluster_data['magnitude'],
                  c=[colors[color_idx]], s=50, alpha=0.6, label=f'C{k}')
        color_idx += 1
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitude')
    ax.set_title('Magnitude Over Time by Cluster')
    ax.legend(fontsize=7, ncol=3)
    
    # 4. Cumulative events by cluster
    ax = axes[1, 1]
    color_idx = 0
    for k in unique_labels:
        if k == -1:
            continue
        mask = labels == k
        cluster_data = plot_df[mask].sort_values('datetime')
        ax.plot(cluster_data['datetime'], range(1, len(cluster_data)+1),
               c=colors[color_idx], linewidth=2, label=f'C{k}')
        color_idx += 1
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Events')
    ax.set_title('Cumulative Event Count by Cluster')
    ax.legend(fontsize=7, ncol=3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_cluster_statistics(df: pd.DataFrame, labels: np.ndarray,
                            title: str = "ST-DBSCAN Cluster Statistics") -> plt.Figure:
    """Plot detailed statistics for each cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plot_df = df.copy()
    plot_df['cluster'] = labels
    
    # Cluster sizes
    ax = axes[0, 0]
    cluster_counts = plot_df['cluster'].value_counts().sort_index()
    colors = ['lightgray' if idx == -1 else plt.cm.tab10(idx % 10) for idx in cluster_counts.index]
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Events')
    ax.set_title('Events per Cluster')
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Mean magnitude by cluster
    ax = axes[0, 1]
    mean_mags = plot_df.groupby('cluster')['magnitude'].mean().sort_index()
    colors = ['lightgray' if idx == -1 else plt.cm.tab10(idx % 10) for idx in mean_mags.index]
    ax.bar(mean_mags.index.astype(str), mean_mags.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean Magnitude')
    ax.set_title('Mean Magnitude by Cluster')
    ax.axhline(y=plot_df['magnitude'].mean(), color='red', linestyle='--',
               label=f'Overall Mean: {plot_df["magnitude"].mean():.2f}')
    ax.legend()
    
    # Depth distribution by cluster
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
    
    # Cluster duration
    ax = axes[1, 1]
    valid_df = plot_df[plot_df['cluster'] != -1]
    durations = valid_df.groupby('cluster')['datetime'].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 86400
    ).sort_index()
    colors = [plt.cm.tab10(idx % 10) for idx in durations.index]
    ax.bar(durations.index.astype(str), durations.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Duration (days)')
    ax.set_title('Cluster Duration (First to Last Event)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_cluster_locations(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """Plot spatial scatter of clusters without cartopy."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    color_idx = 0
    for k in unique_labels:
        mask = labels == k
        
        if k == -1:
            ax.scatter(df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                      c='lightgray', s=15, alpha=0.3, marker='x',
                      label=f'Noise ({mask.sum()})')
        else:
            sizes = df.loc[mask, 'magnitude'] ** 2 * 8
            ax.scatter(df.loc[mask, 'longitude'], df.loc[mask, 'latitude'],
                      c=[colors[color_idx]], s=sizes, alpha=0.6, edgecolors='black', linewidths=0.3,
                      label=f'Cluster {k} ({mask.sum()})')
            color_idx += 1
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('ST-DBSCAN Spatial Clusters')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_stdbscan_analysis(filepath: str,
                          min_magnitude: float = 2.0,
                          spatial_eps_km: float = 50.0,
                          temporal_eps_days: float = 7.0,
                          min_samples: int = 5,
                          use_optimized: bool = True,
                          output_dir: str = "stdbscan_results") -> Dict:
    """
    Run the complete ST-DBSCAN clustering analysis pipeline.
    
    Parameters
    ----------
    filepath : str
        Path to earthquake data CSV
    min_magnitude : float
        Minimum magnitude threshold
    spatial_eps_km : float
        Spatial epsilon in kilometers
    temporal_eps_days : float
        Temporal epsilon in days
    min_samples : int
        Minimum samples for core point
    use_optimized : bool
        Use optimized BallTree implementation (faster)
    output_dir : str
        Directory to save outputs
    
    Returns
    -------
    Dict
        Results including data, labels, and metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_earthquake_data(filepath)
    df = apply_magnitude_threshold(df, min_magnitude)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Run ST-DBSCAN
    print("\n" + "=" * 60)
    print("RUNNING ST-DBSCAN CLUSTERING")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Spatial epsilon: {spatial_eps_km} km")
    print(f"  - Temporal epsilon: {temporal_eps_days} days")
    print(f"  - Min samples: {min_samples}")
    print(f"  - Using {'optimized (BallTree)' if use_optimized else 'naive O(n²)'} implementation")
    
    if use_optimized:
        labels = run_st_dbscan_optimized(df, spatial_eps_km, temporal_eps_days, min_samples)
    else:
        labels = run_st_dbscan_naive(df, spatial_eps_km, temporal_eps_days, min_samples)
    
    # 4. Compute metrics
    metrics = compute_clustering_metrics(df, labels)
    cluster_stats = compute_cluster_characteristics(df, labels)
    
    print("\nClustering Metrics:")
    print(f"  - Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"  - Calinski-Harabasz: {metrics['calinski_harabasz']:.4f}")
    print(f"  - Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    print(f"  - Number of clusters: {metrics['n_clusters']}")
    print(f"  - Noise points: {metrics['n_noise']} ({metrics['noise_ratio']*100:.1f}%)")
    
    # 5. Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Philippine map
    fig = plot_clusters_on_map(df, labels)
    if fig is not None:
        fig.savefig(f"{output_dir}/stdbscan_map.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Simple spatial scatter (backup if cartopy not available)
    fig = plot_cluster_locations(df, labels)
    fig.savefig(f"{output_dir}/spatial_clusters.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Spatiotemporal view
    fig = plot_spatiotemporal_view(df, labels)
    fig.savefig(f"{output_dir}/spatiotemporal_view.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Cluster statistics
    fig = plot_cluster_statistics(df, labels)
    fig.savefig(f"{output_dir}/cluster_statistics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    df['cluster_stdbscan'] = labels
    df.to_csv(f"{output_dir}/clustered_earthquakes.csv", index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/clustering_metrics.csv", index=False)
    
    # Save cluster characteristics
    cluster_stats.to_csv(f"{output_dir}/cluster_characteristics.csv", index=False)
    
    print(f"Results saved to {output_dir}/")
    
    return {
        'data': df,
        'labels': labels,
        'metrics': metrics,
        'cluster_stats': cluster_stats
    }


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_FILE = "data/earthquake_data_20260203_184010.csv"
    OUTPUT_DIR = "stdbscan_results"
    
    # Run analysis
    results = run_stdbscan_analysis(
        filepath=DATA_FILE,
        min_magnitude=2.0,
        spatial_eps_km=50.0,      # Events within 50km
        temporal_eps_days=7.0,     # Events within 7 days
        min_samples=5,
        use_optimized=True,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "=" * 60)
    print("ST-DBSCAN ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - stdbscan_map.png")
    print("  - spatial_clusters.png")
    print("  - spatiotemporal_view.png")
    print("  - cluster_statistics.png")
    print("  - clustering_metrics.csv")
    print("  - cluster_characteristics.csv")
    print("  - clustered_earthquakes.csv")
    
    # Print cluster summary
    print("\nCluster Summary:")
    print(results['cluster_stats'].to_string(index=False))
