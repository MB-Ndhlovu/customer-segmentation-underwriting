"""KMeans clustering, Elbow method, Silhouette analysis, and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> dict:
    """Run Elbow + Silhouette analysis across k_range; return best k and metrics."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(0.0)

    # Choose k with highest silhouette, capped at reasonable range
    silhouette_by_k = {k: s for k, s in zip(k_range, silhouettes) if k > 1}
    best_k = max(silhouette_by_k, key=silhouette_by_k.get)

    return {
        "best_k": int(best_k),
        "inertias": {k: float(i) for k, i in zip(k_range, inertias)},
        "silhouettes": {k: float(s) for k, s in zip(k_range, silhouettes)},
        "best_silhouette": float(silhouette_by_k[best_k]),
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans with n_clusters; return model + labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Build per-segment profile summary (mean of each feature)."""
    df_tmp = df.copy()
    df_tmp["_cluster"] = labels
    profiles = df_tmp.groupby("_cluster")[feature_cols].mean().round(2)
    profiles["count"] = df_tmp.groupby("_cluster").size()
    return profiles.reset_index().rename(columns={"_cluster": "cluster"})


def assign_segment_names(df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Assign human-readable risk tier names to cluster labels based on profile means."""
    df_tmp = df.copy()
    df_tmp["_cluster"] = labels

    seg_order = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]

    # Order clusters by income descending then DTI ascending to map to names
    profile = (
        df_tmp.groupby("_cluster")[["income", "debt_to_income", "credit_score"]]
        .mean()
        .sort_values(["income", "credit_score"], ascending=[False, False])
    )
    cluster_to_name = {
        profile.index[0]: "Established Prime",
        profile.index[1]: "Rising Prime",
        profile.index[2]: "Mass Market",
        profile.index[3]: "Subprime High-Risk",
    }

    names = pd.Series(labels).map(cluster_to_name)
    return names