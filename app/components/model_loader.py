"""
Model loader component for multi-version HDBSCAN model management.

Discovers and loads HDBSCAN clustering models from app/models/<version>/model.joblib.
"""

import os
import json
import joblib
from typing import Dict, List, Optional


# Path to the models directory (relative to project root)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def discover_model_versions() -> List[str]:
    """
    Discover all available model versions in app/models/.

    Looks for directories like v1/, v2/, etc. containing a model.joblib file.

    Returns
    -------
    List[str]
        Sorted list of version names (e.g. ['v1', 'v2'])
    """
    if not os.path.exists(MODELS_DIR):
        return []

    versions = []
    for entry in sorted(os.listdir(MODELS_DIR)):
        version_dir = os.path.join(MODELS_DIR, entry)
        model_path = os.path.join(version_dir, "model.joblib")
        if os.path.isdir(version_dir) and os.path.exists(model_path):
            versions.append(entry)

    return versions


def load_model(version: str) -> Dict:
    """
    Load a specific model version.

    Parameters
    ----------
    version : str
        Version name (e.g. 'v1')

    Returns
    -------
    Dict
        Model data containing scaler, hyperparameters, feature_columns,
        metrics, labels, and the HDBSCAN clusterer.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist
    """
    model_path = os.path.join(MODELS_DIR, version, "model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = joblib.load(model_path)
    return model_data


def get_model_info(model_data: Dict) -> Dict:
    """
    Extract summary information from a loaded model (HDBSCAN only).

    Parameters
    ----------
    model_data : Dict
        Loaded model data

    Returns
    -------
    Dict
        Summary with timestamp, feature count, HDBSCAN metrics, hyperparameters
    """
    info = {
        "timestamp": model_data.get("timestamp", "unknown"),
        "feature_columns": model_data.get("feature_columns", []),
        "n_features": len(model_data.get("feature_columns", [])),
        "has_scaler": "scaler" in model_data,
    }

    # Extract HDBSCAN hyperparameters
    hyperparams = model_data.get("hyperparameters", {})
    hdbscan_params = model_data.get("hdbscan_best_params", {})
    info["hyperparameters"] = hyperparams
    info["hdbscan_best_params"] = hdbscan_params

    # Extract only HDBSCAN metrics
    all_metrics = model_data.get("metrics", {})
    hdbscan_metrics = {}
    for key, value in all_metrics.items():
        if "hdbscan" in key.lower():
            hdbscan_metrics[key] = value
    info["metrics"] = hdbscan_metrics

    return info


def get_scaler(model_data: Dict):
    """
    Extract the fitted StandardScaler from model data.

    Returns
    -------
    StandardScaler or None
    """
    return model_data.get("scaler", None)


def get_hdbscan_clusterer(model_data: Dict):
    """
    Extract the fitted HDBSCAN clusterer from model data.

    Returns
    -------
    HDBSCAN clusterer object or None
    """
    # Try direct key first
    clusterer = model_data.get("hdbscan_clusterer", None)
    if clusterer is not None:
        return clusterer

    # Fallback: look in grid_search_results
    gs = model_data.get("grid_search_results", {})
    if isinstance(gs, dict) and "hdbscan" in gs:
        return gs["hdbscan"].get("best_clusterer", None)

    return None


def get_feature_columns(model_data: Dict) -> List[str]:
    """
    Get the feature column names the model was trained with.

    Returns
    -------
    List[str]
        Feature column names
    """
    return model_data.get("feature_columns", [])


def load_cluster_labels(version: str) -> Optional[Dict]:
    """
    Load cluster label descriptions for a specific model version.

    Looks for a cluster_labels.json file in the model version directory.

    Parameters
    ----------
    version : str
        Version name (e.g. 'v2')

    Returns
    -------
    Dict or None
        Cluster labels data with 'clusters' and 'noise' keys,
        or None if no labels file exists.
    """
    labels_path = os.path.join(MODELS_DIR, version, "cluster_labels.json")

    if not os.path.exists(labels_path):
        return None

    with open(labels_path, "r") as f:
        return json.load(f)
