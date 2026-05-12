"""KMeans clustering with Elbow method, Silhouette analysis, and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> dict:
    """Run Elbow + Silhouette analysis across k_range."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return {"inertias": inertias, "silhouettes": silhouettes, "k_values": list(k_range)}


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> tuple:
    """Fit KMeans with n_clusters; return labels + silhouette."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return labels, km.cluster_centers_, sil


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build centroid-style profile table per segment."""
    df = df.copy()
    df["segment_label"] = labels
    profiles = df.groupby("segment_label").mean(numeric_only=True)
    profiles["count"] = df.groupby("segment_label").size()
    profiles["segment_name"] = profiles.index.map(SEGMENT_NAMES)
    return profiles.reset_index()


def remap_clusters(profiles: pd.DataFrame) -> dict:
    """
    Remap KMeans cluster IDs to business-meaningful names based on
    credit_score and income medians.
    """
    medians = profiles[["credit_score", "income", "segment_label"]].set_index("segment_label")
    sorted_cs = medians["credit_score"].sort_values().index.tolist()
    sorted_inc = medians["income"].sort_values().index.tolist()

    mapping = {}
    mapping[sorted_cs[0]] = "Subprime High-Risk"
    mapping[sorted_cs[-1]] = "Established Prime"

    mid = sorted_cs[1] if len(sorted_cs) > 2 else sorted_cs[1]
    mapping[sorted_cs[1]] = "Mass Market" if medians.loc[sorted_cs[1], "income"] < medians.loc[sorted_cs[-2], "income"] else "Rising Prime"

    if len(sorted_cs) == 4:
        mapping[sorted_cs[2]] = "Rising Prime" if mapping.get(sorted_cs[1]) == "Mass Market" else "Mass Market"

    for k, v in SEGMENT_NAMES.items():
        if k not in mapping:
            mapping[k] = v
    return mapping