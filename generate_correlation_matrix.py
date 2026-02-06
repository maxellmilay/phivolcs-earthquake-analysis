"""
Generate a correlation matrix heatmap for the 10 selected clustering features
using the clustered_earthquakes.csv output file.

Usage:
    python generate_correlation_matrix.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
INPUT_CSV = "clustering_results_100k_datapoints/clustered_earthquakes.csv"
OUTPUT_PNG = "clustering_results_100k_datapoints/selected_features_correlation_matrix.png"

FEATURES = [
    # Core Physical
    "magnitude",
    "depth",
    # Magnitude Dynamics
    "mag_diff_from_prev",
    "mag_rolling_std_50",
    # Spatial Activity
    "local_density",
    "events_last_7d",
    "events_last_30d",
    # Temporal Context
    "hrs_since_significant_event",
    "global_interevent_hrs",
]

SHORT_LABELS = [
    "magnitude",
    "depth",
    "mag_diff_prev",
    "mag_std_50",
    "local_density",
    "events_7d",
    "events_30d",
    "hrs_since_sig",
    "interevent_hrs",
]

# --------------------------------------------------------------------------
# LOAD & COMPUTE
# --------------------------------------------------------------------------
print(f"Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df)} rows loaded")

subset = df[FEATURES]
corr = subset.corr()

# --------------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

cmap = sns.diverging_palette(250, 15, s=75, l=40, n=256, center="light")

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.8,
    linecolor="white",
    xticklabels=SHORT_LABELS,
    yticklabels=SHORT_LABELS,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax,
)

ax.set_title(
    "Feature Correlation Matrix (10 Selected Clustering Features)",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"  Saved: {OUTPUT_PNG}")
plt.close(fig)

# --------------------------------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------------------------------
print("\n=== Highly Correlated Pairs (|r| >= 0.5) ===")
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.5:
            print(f"  {FEATURES[i]:30s} <-> {FEATURES[j]:30s}  r = {r:+.4f}")

print("\n=== Moderately Correlated Pairs (0.3 <= |r| < 0.5) ===")
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        r = corr.iloc[i, j]
        if 0.3 <= abs(r) < 0.5:
            print(f"  {FEATURES[i]:30s} <-> {FEATURES[j]:30s}  r = {r:+.4f}")

print("\nDone.")
