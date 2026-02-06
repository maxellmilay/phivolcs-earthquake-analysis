"""
Preprocessing component for earthquake clustering inference.

Handles feature selection, scaling, and NaN imputation,
matching clustering_experiment.py exactly.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

# The 15-feature set used in clustering_experiment.py
LOCATION_AGNOSTIC_FEATURES = [
    "magnitude",
    "depth",
    "magnitude_zscore",
    "depth_zscore",
    "is_deep",
    "is_significant",
    "is_major",
    "mag_deviation_from_recent",
    "mag_diff_from_prev",
    "mag_rolling_std_50",
    "local_density",
    "events_last_7d",
    "events_last_30d",
    "hrs_since_significant_event",
    "global_interevent_hrs",
]


def select_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Select the location-agnostic features used for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all engineered features
    feature_cols : List[str], optional
        Feature columns to use. Defaults to LOCATION_AGNOSTIC_FEATURES.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the selected feature columns
    """
    if feature_cols is None:
        feature_cols = LOCATION_AGNOSTIC_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    return df[available]


def impute_nans(X: np.ndarray) -> np.ndarray:
    """
    Impute NaN values with column medians (matching training pipeline).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix potentially containing NaNs

    Returns
    -------
    np.ndarray
        Feature matrix with NaNs replaced by column medians
    """
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        col_medians = np.nanmedian(X, axis=0)
        # Replace NaN medians with 0 (edge case for all-NaN columns)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_medians, nan_indices[1])
    return X


def scale_features(
    X: np.ndarray, scaler: StandardScaler = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.

    If a fitted scaler is provided (from the saved model), it uses that.
    Otherwise fits a new one (should not happen in inference).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    scaler : StandardScaler, optional
        Pre-fitted scaler from the trained model

    Returns
    -------
    Tuple[np.ndarray, StandardScaler]
        (scaled features, scaler used)
    """
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def prepare_for_inference(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Full preprocessing pipeline for inference: select features, impute, scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all engineered features
    scaler : StandardScaler
        Pre-fitted scaler from the trained model
    feature_cols : List[str], optional
        Feature columns to use. Defaults to LOCATION_AGNOSTIC_FEATURES.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (scaled feature matrix, list of feature columns used)
    """
    if feature_cols is None:
        feature_cols = LOCATION_AGNOSTIC_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.copy()

    X = impute_nans(X)
    X_scaled, _ = scale_features(X, scaler)

    return X_scaled, available
