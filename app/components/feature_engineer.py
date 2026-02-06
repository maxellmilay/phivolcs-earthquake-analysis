"""
Feature engineering component for earthquake clustering inference.

These transformations are exact replicas of those in clustering_experiment.py
to ensure consistency between training and inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.neighbors import BallTree


# =============================================================================
# SPATIAL FEATURES
# =============================================================================

def compute_local_quake_density(df: pd.DataFrame, radius_km: float = 50.0) -> np.ndarray:
    """
    Compute local quake density using BallTree (haversine metric).

    Counts the number of other events within `radius_km` of each event.
    """
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    tree = BallTree(coords_rad, metric="haversine")
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True) - 1

    return counts


def compute_interevent_features(
    df: pd.DataFrame, radius_km: float = 25.0, time_window_days: float = 30.0
):
    """
    Compute interevent time and distance to nearest previous event
    within a spatial radius using BallTree + temporal pruning.
    """
    n = len(df)
    coords = np.radians(df[["latitude", "longitude"]].values)
    times = df["datetime"].values

    time_since_prev = np.full(n, np.nan)
    dist_to_prev = np.full(n, np.nan)

    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM
    time_window_ns = np.timedelta64(int(time_window_days * 86400), "s")

    for i in range(1, n):
        mask_time = times[:i] >= times[i] - time_window_ns
        if not np.any(mask_time):
            continue

        prev_coords = coords[:i][mask_time]
        sub_tree = BallTree(prev_coords, metric="haversine")
        query_point = coords[i].reshape(1, -1)
        ind = sub_tree.query_radius(query_point, r=radius_rad, return_distance=True)
        distances_rad, indices = ind[1][0], ind[0][0]

        if len(indices) == 0:
            continue

        distances_km = distances_rad * EARTH_RADIUS_KM
        recent_times = times[:i][mask_time][indices]
        valid_mask = recent_times < times[i]

        if not np.any(valid_mask):
            continue

        deltas = (times[i] - recent_times[valid_mask]) / np.timedelta64(1, "h")
        nearest_idx = np.argmin(deltas)
        time_since_prev[i] = deltas[nearest_idx]
        dist_to_prev[i] = distances_km[valid_mask][nearest_idx]

    return time_since_prev, dist_to_prev


def compute_temporal_density(
    df: pd.DataFrame,
    radius_km: float = 50.0,
    windows_days: List[int] = [1, 7, 30],
    max_window_days: float = 30.0,
) -> Dict[int, np.ndarray]:
    """
    Compute the number of nearby events within each temporal window
    using spatial BallTree queries + temporal pruning.
    """
    n = len(df)
    coords = np.radians(df[["latitude", "longitude"]].values)
    times = df["datetime"].values.astype("datetime64[ns]")

    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    results = {w: np.zeros(n, dtype=int) for w in windows_days}
    max_window = np.timedelta64(int(max_window_days * 86400), "s")

    for i in range(n):
        t_i = times[i]
        mask_time = times[:i] >= t_i - max_window
        if not np.any(mask_time):
            continue

        prev_coords = coords[:i][mask_time]
        prev_times = times[:i][mask_time]

        if len(prev_coords) == 0:
            continue

        sub_tree = BallTree(prev_coords, metric="haversine")
        query_point = coords[i].reshape(1, -1)
        ind = sub_tree.query_radius(query_point, r=radius_rad, return_distance=False)

        if len(ind[0]) == 0:
            continue

        candidate_times = prev_times[ind[0]]
        deltas_days = (t_i - candidate_times) / np.timedelta64(1, "D")

        for w in windows_days:
            results[w][i] = np.sum(deltas_days <= w)

    return results


# =============================================================================
# CORE FEATURE ENGINEERING PIPELINE
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    density_radius_km: float = 50.0,
    interevent_radius_km: float = 25.0,
    temporal_windows: List[int] = [1, 7, 30],
    rolling_windows: List[int] = [10, 50, 100],
) -> pd.DataFrame:
    """
    Apply all feature engineering steps, matching clustering_experiment.py exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake data with columns: datetime, latitude, longitude, depth, magnitude
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
        Data with all engineered features
    """
    result = df.copy()

    # 1. Temporal features
    result["hour"] = result["datetime"].dt.hour
    result["day_of_week"] = result["datetime"].dt.dayofweek
    result["day_of_year"] = result["datetime"].dt.dayofyear
    result["month"] = result["datetime"].dt.month
    result["days_since_start"] = (
        result["datetime"] - result["datetime"].min()
    ).dt.total_seconds() / 86400

    # 2. Energy features (Gutenberg-Richter: log10(E) = 1.5*M + 4.8)
    result["log_energy"] = 1.5 * result["magnitude"] + 4.8
    reference_energy = 1.5 * 2.0 + 4.8
    result["energy_ratio_log"] = result["log_energy"] - reference_energy

    # 3. Rolling magnitude statistics
    for w in rolling_windows:
        result[f"mag_rolling_mean_{w}"] = (
            result["magnitude"].rolling(window=w, min_periods=1).mean()
        )
        result[f"mag_rolling_std_{w}"] = (
            result["magnitude"].rolling(window=w, min_periods=1).std().fillna(0)
        )
        result[f"mag_rolling_max_{w}"] = (
            result["magnitude"].rolling(window=w, min_periods=1).max()
        )
    result["mag_deviation_from_recent"] = (
        result["magnitude"] - result["mag_rolling_mean_50"]
    )

    # 4. Depth features
    result["is_shallow"] = (result["depth"] <= 70).astype(int)
    result["is_intermediate"] = (
        (result["depth"] > 70) & (result["depth"] <= 300)
    ).astype(int)
    result["is_deep"] = (result["depth"] > 300).astype(int)
    result["log_depth"] = np.log1p(result["depth"])
    depth_mean = result["depth"].mean()
    depth_std = result["depth"].std()
    if depth_std == 0:
        depth_std = 1.0
    result["depth_zscore"] = (result["depth"] - depth_mean) / depth_std

    # 5. Magnitude anomaly features
    mag_mean = result["magnitude"].mean()
    mag_std = result["magnitude"].std()
    if mag_std == 0:
        mag_std = 1.0
    result["magnitude_zscore"] = (result["magnitude"] - mag_mean) / mag_std
    result["magnitude_percentile"] = result["magnitude"].rank(pct=True)
    result["is_significant"] = (result["magnitude"] >= 5.0).astype(int)
    result["is_major"] = (result["magnitude"] >= 6.0).astype(int)

    # 6. Cyclical encoding
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    result["doy_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365)
    result["doy_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365)

    # 7. Global temporal features
    n = len(result)
    time_diffs = result["datetime"].diff()
    result["global_interevent_hrs"] = time_diffs.dt.total_seconds() / 3600
    result["global_interevent_hrs"] = result["global_interevent_hrs"].fillna(
        result["global_interevent_hrs"].median()
    )
    result["log_interevent_hrs"] = np.log1p(result["global_interevent_hrs"])

    # Time since last significant event (M >= 5.0)
    significant_mask = result["magnitude"] >= 5.0
    time_since_significant = np.full(n, np.nan)
    last_significant_time = None
    for i in range(n):
        if last_significant_time is not None:
            time_diff = (
                result["datetime"].iloc[i] - last_significant_time
            ).total_seconds() / 3600
            time_since_significant[i] = time_diff
        if significant_mask.iloc[i]:
            last_significant_time = result["datetime"].iloc[i]

    result["hrs_since_significant_event"] = time_since_significant
    result["hrs_since_significant_event"] = result[
        "hrs_since_significant_event"
    ].fillna(result["hrs_since_significant_event"].max())

    # Rolling event rates
    time_diffs_days = result["datetime"].diff().dt.total_seconds() / 86400
    time_diffs_days = time_diffs_days.fillna(0)
    for window in [20, 50, 100]:
        time_spans_days = time_diffs_days.rolling(window=window, min_periods=1).sum()
        result[f"event_rate_last_{window}"] = window / time_spans_days.replace(
            0, np.nan
        )
        result[f"event_rate_last_{window}"] = result[
            f"event_rate_last_{window}"
        ].fillna(result[f"event_rate_last_{window}"].median())

    # 8. Sequence features
    result["mag_diff_from_prev"] = result["magnitude"].diff().fillna(0)
    result["cumulative_mag_sum"] = result["magnitude"].cumsum()
    result["event_number"] = range(1, n + 1)
    result["catalog_progress"] = result["event_number"] / n
    result["mag_below_recent_max"] = (
        result["mag_rolling_max_50"] - result["magnitude"]
    )

    # 9. Local quake density
    result["local_density"] = compute_local_quake_density(result, density_radius_km)

    # 10. Interevent features
    time_since, dist_to = compute_interevent_features(result, interevent_radius_km)
    result["time_since_prev_event_hrs"] = time_since
    result["dist_to_prev_event_km"] = dist_to

    # 11. Temporal density
    temporal_densities = compute_temporal_density(
        result, density_radius_km, temporal_windows
    )
    for w, counts in temporal_densities.items():
        result[f"events_last_{w}d"] = counts

    # Fill NaN values with median
    nan_fill_cols = [
        "time_since_prev_event_hrs",
        "dist_to_prev_event_km",
        "hrs_since_significant_event",
    ]
    for col in nan_fill_cols:
        if col in result.columns:
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)

    return result
