"""
Earthquake Clustering Streamlit App
====================================

Displays the latest seismic events overlayed on a Philippine map,
colored by their predicted HDBSCAN cluster from a trained model.

Supports multiple model versions from app/models/<version>/model.joblib.

Run with:
    streamlit run app/entrypoint.py
"""

import sys
import os

# Ensure the project root is on the path so component imports work
# regardless of where streamlit is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from typing import Optional, Dict

from components.scraper import scrape_latest
from components.feature_engineer import engineer_features
from components.preprocess import prepare_for_inference, LOCATION_AGNOSTIC_FEATURES
from components.model_loader import (
    discover_model_versions,
    load_model,
    get_model_info,
    get_scaler,
    get_hdbscan_clusterer,
    get_feature_columns,
    load_cluster_labels,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="PH Earthquake Cluster Monitor",
    page_icon="🌍",
    layout="wide",
)

# =============================================================================
# CONSTANTS
# =============================================================================

PH_LAT_MIN, PH_LAT_MAX = 4.0, 22.0
PH_LON_MIN, PH_LON_MAX = 116.0, 128.0

# Cluster color palette (distinct, high-contrast colors)
CLUSTER_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#e377c2",  # rose
]
NOISE_COLOR = "#999999"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def predict_clusters(model_data: dict, X_scaled: np.ndarray) -> np.ndarray:
    """Predict cluster labels using the HDBSCAN model."""
    clusterer = get_hdbscan_clusterer(model_data)
    if clusterer is not None and hasattr(clusterer, "prediction_data_"):
        try:
            labels, _ = hdbscan.approximate_predict(clusterer, X_scaled)
            return labels
        except Exception as e:
            st.error(f"HDBSCAN approximate_predict failed: {e}")

    st.error(
        "No HDBSCAN clusterer with prediction data found in the model. "
        "Please retrain the model with `prediction_data=True`."
    )
    return np.full(len(X_scaled), -1)


def get_cluster_color(label: int) -> str:
    """Get color for a cluster label."""
    if label == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


def get_cluster_label(cluster_id: int, cluster_labels: Optional[Dict]) -> str:
    """Get the human-readable label for a cluster from JSON, or fallback to generic name."""
    if cluster_labels is None:
        return "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    
    if cluster_id == -1:
        return cluster_labels.get("noise", {}).get("label", "Noise")
    
    cluster_key = str(cluster_id)
    clusters = cluster_labels.get("clusters", {})
    if cluster_key in clusters:
        return clusters[cluster_key].get("label", f"Cluster {cluster_id}")
    
    return f"Cluster {cluster_id}"


def get_cluster_description(cluster_id: int, cluster_labels: Optional[Dict]) -> str:
    """Get the description for a cluster from JSON, or return empty string."""
    if cluster_labels is None:
        return ""
    
    if cluster_id == -1:
        return cluster_labels.get("noise", {}).get("description", "")
    
    cluster_key = str(cluster_id)
    clusters = cluster_labels.get("clusters", {})
    if cluster_key in clusters:
        return clusters[cluster_key].get("description", "")
    
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def fetch_earthquake_data(max_rows: int = 100) -> pd.DataFrame:
    """Scrape latest earthquake data (cached for 5 minutes)."""
    return scrape_latest(max_rows=max_rows)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("Philippine Earthquake Cluster Monitor")
    st.markdown(
        "Real-time seismic activity clustering using HDBSCAN with "
        "location-agnostic features."
    )

    # -----------------------------------------------------------------
    # Sidebar: Model Version Selection
    # -----------------------------------------------------------------
    st.sidebar.header("Model Configuration")

    versions = discover_model_versions()
    if not versions:
        st.error(
            "No model versions found in `app/models/`. "
            "Please train a model first using `clustering_experiment.py`."
        )
        return

    selected_version = st.sidebar.selectbox(
        "Model Version",
        versions,
        index=len(versions) - 1,  # Default to latest
        help="Select which trained HDBSCAN model version to use for inference.",
    )

    # Load model
    with st.spinner(f"Loading model {selected_version}..."):
        model_data = load_model(selected_version)

    scaler = get_scaler(model_data)
    feature_cols = get_feature_columns(model_data)
    model_info = get_model_info(model_data)
    cluster_labels = load_cluster_labels(selected_version)

    # Display model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.text(f"Version: {selected_version}")
    st.sidebar.text(f"Trained: {model_info['timestamp']}")
    st.sidebar.text(f"Features: {model_info['n_features']}")

    # Show HDBSCAN best params if available
    hdbscan_params = model_info.get("hdbscan_best_params", {})
    if hdbscan_params:
        st.sidebar.markdown("**HDBSCAN Parameters**")
        for param, value in hdbscan_params.items():
            st.sidebar.text(f"  {param}: {value}")

    # Show only HDBSCAN training metrics
    if model_info["metrics"]:
        st.sidebar.markdown("**HDBSCAN Metrics (training)**")
        for algo, metrics in model_info["metrics"].items():
            if isinstance(metrics, dict):
                sil = metrics.get("silhouette", None)
                ch = metrics.get("calinski_harabasz", None)
                db = metrics.get("davies_bouldin", None)
                n_clusters = metrics.get("n_clusters", None)
                if n_clusters is not None:
                    st.sidebar.text(f"  Clusters: {n_clusters}")
                if isinstance(sil, float):
                    st.sidebar.text(f"  Silhouette: {sil:.4f}")
                if isinstance(ch, float):
                    st.sidebar.text(f"  Calinski-Harabasz: {ch:.2f}")
                if isinstance(db, float):
                    st.sidebar.text(f"  Davies-Bouldin: {db:.4f}")

    if not scaler:
        st.error("Model is missing a fitted scaler. Cannot run inference.")
        return

    if not feature_cols:
        st.warning(
            "Model did not store feature_columns. "
            "Falling back to default 15-feature set."
        )
        feature_cols = LOCATION_AGNOSTIC_FEATURES

    # -----------------------------------------------------------------
    # Sidebar: Display Settings
    # -----------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Settings")

    n_display = st.sidebar.slider(
        "Number of events to display",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="How many of the latest seismic events to show on the map.",
    )

    # -----------------------------------------------------------------
    # Sidebar: Data Source
    # -----------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Source")

    data_source = st.sidebar.radio(
        "Source",
        ["Live (PHIVOLCS)", "Upload CSV"],
        help="Fetch live data from PHIVOLCS or upload a local CSV file.",
    )

    df_raw = None

    if data_source == "Live (PHIVOLCS)":
        # Auto-scrape on page load
        if "df_raw" not in st.session_state:
            with st.spinner("Scraping latest data from PHIVOLCS..."):
                try:
                    df_raw = fetch_earthquake_data(max_rows=100)
                    st.session_state["df_raw"] = df_raw
                except Exception as e:
                    st.error(f"Failed to scrape data: {e}")

        # Allow manual refresh
        if st.sidebar.button("Refresh Data", type="primary"):
            with st.spinner("Scraping latest data from PHIVOLCS..."):
                try:
                    fetch_earthquake_data.clear()
                    df_raw = fetch_earthquake_data(max_rows=100)
                    st.session_state["df_raw"] = df_raw
                except Exception as e:
                    st.error(f"Failed to scrape data: {e}")

        if "df_raw" in st.session_state:
            df_raw = st.session_state["df_raw"]
    else:
        uploaded = st.sidebar.file_uploader("Upload earthquake CSV", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            if "datetime" in df_raw.columns:
                df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
            for col in ["latitude", "longitude", "depth", "magnitude"]:
                if col in df_raw.columns:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw = df_raw.dropna(subset=["latitude", "longitude", "datetime", "magnitude"])
            df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
            if len(df_raw) > 100:
                df_raw = df_raw.tail(100).reset_index(drop=True)

    if df_raw is None or len(df_raw) == 0:
        st.info(
            "Waiting for data... If auto-scrape failed, try clicking "
            "**Refresh Data** in the sidebar or upload a CSV file."
        )
        return

    st.sidebar.success(f"Loaded {len(df_raw)} records")

    # -----------------------------------------------------------------
    # Feature Engineering & Inference
    # -----------------------------------------------------------------
    with st.spinner("Engineering features..."):
        df_features = engineer_features(df_raw)

    with st.spinner("Running HDBSCAN cluster inference..."):
        X_scaled, used_features = prepare_for_inference(
            df_features, scaler, feature_cols
        )
        labels = predict_clusters(model_data, X_scaled)

    df_features["cluster"] = labels
    # Add cluster label column for display
    df_features["cluster_label"] = df_features["cluster"].apply(
        lambda c: get_cluster_label(c, cluster_labels)
    )

    # Take top N latest events based on slider
    df_latest = df_features.sort_values("datetime", ascending=False).head(n_display).copy()
    df_latest = df_latest.sort_values("datetime").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Map Visualization
    # -----------------------------------------------------------------
    st.header(f"Top {n_display} Latest Seismic Events by Cluster")

    # Assign colors
    df_latest["color"] = df_latest["cluster"].apply(get_cluster_color)

    # Build legend HTML items (with hover descriptions, no counts)
    unique_clusters = sorted(df_latest["cluster"].unique())
    legend_html_items = ""
    for c in unique_clusters:
        label = get_cluster_label(c, cluster_labels)
        desc = get_cluster_description(c, cluster_labels)
        color = get_cluster_color(c)
        title_attr = f' title="{desc}"' if desc else ""
        legend_html_items += (
            f'<div class="legend-item"{title_attr}>'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'<span class="legend-label">{label}</span>'
            f'</div>'
        )

    # Pydeck map with overlaid legend
    try:
        import pydeck as pdk

        hex_to_rgb = lambda h: [int(h[i:i+2], 16) for i in (1, 3, 5)]
        df_latest["color_rgb"] = df_latest["color"].apply(hex_to_rgb)

        # Scale size by magnitude
        df_latest["radius"] = df_latest["magnitude"] ** 2 * 2000

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_latest,
            get_position=["longitude", "latitude"],
            get_radius="radius",
            get_fill_color="color_rgb",
            opacity=0.7,
            pickable=True,
            auto_highlight=True,
        )

        view = pdk.ViewState(
            latitude=12.5,
            longitude=122.0,
            zoom=5,
            pitch=0,
        )

        tooltip = {
            "html": (
                "<b>Cluster:</b> {cluster_label}<br>"
                "<b>Magnitude:</b> {magnitude}<br>"
                "<b>Depth:</b> {depth} km<br>"
                "<b>Location:</b> {location}<br>"
                "<b>Date:</b> {datetime}"
            ),
            "style": {"backgroundColor": "#333", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        )

        # Display legend directly above map (clearly visible)
        st.markdown(
            f"""
            <style>
            .map-legend-box {{
                background: #ffffff;
                border: 2px solid #333333;
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 12px;
                display: inline-block;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-size: 14px;
                line-height: 1.8;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 5px 0;
                cursor: help;
                white-space: nowrap;
            }}
            .legend-item[title]:hover {{
                background: rgba(0,0,0,0.08);
                border-radius: 4px;
                padding-left: 6px;
                padding-right: 6px;
            }}
            .legend-dot {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                display: inline-block;
                flex-shrink: 0;
                border: 2px solid rgba(0,0,0,0.2);
            }}
            .legend-label {{
                color: #1a1a1a;
                font-weight: 600;
                font-size: 14px;
            }}
            </style>
            <div class="map-legend-box">
                <strong style="color: #333; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Legend</strong>
                <div style="margin-top: 8px;">
                    {legend_html_items}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Render the map
        st.pydeck_chart(deck, use_container_width=True)

    except ImportError:
        st.warning("Install `pydeck` for better map visualization. Falling back to basic map.")
        st.map(df_latest[["latitude", "longitude"]])

    # -----------------------------------------------------------------
    # Data Table
    # -----------------------------------------------------------------
    st.header("Event Details")

    display_cols = ["datetime", "latitude", "longitude", "depth", "magnitude", "cluster_label"]
    if "location" in df_latest.columns:
        display_cols.insert(5, "location")

    st.dataframe(
        df_latest[display_cols].sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------
    # Cluster Summary
    # -----------------------------------------------------------------
    st.header("Cluster Summary")

    cluster_summary = (
        df_latest.groupby("cluster")
        .agg(
            count=("magnitude", "size"),
            mean_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            mean_depth=("depth", "mean"),
        )
        .round(2)
        .reset_index()
    )
    cluster_summary["cluster"] = cluster_summary["cluster"].apply(
        lambda c: get_cluster_label(c, cluster_labels)
    )
    cluster_summary.columns = ["Cluster", "Count", "Mean Mag", "Max Mag", "Mean Depth (km)"]

    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
