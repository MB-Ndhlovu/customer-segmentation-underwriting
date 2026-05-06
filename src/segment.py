import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_optimal_k(X_scaled: np.ndarray, k_range: range, random_state: int = 42) -> dict:
    """Elbow method + silhouette analysis to pick best k."""
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = k_range[np.argmax(silhouettes)]
    return {"best_k": best_k, "inertias": inertias, "silhouettes": silhouettes, "k_values": list(k_range)}


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_names: list) -> dict:
    """Profile each cluster with feature means and assign business labels."""
    df_labeled = df.copy()
    df_labeled["cluster"] = labels

    segment_names = {
        0: "Mass Market",
        1: "Rising Prime",
        2: "Established Prime",
        3: "Subprime High-Risk",
    }

    # Compute cluster centroids in original feature space
    centroids = df_labeled.groupby("cluster")[feature_names].mean()

    profiles = {}
    for cluster_id in sorted(df_labeled["cluster"].unique()):
        mask = labels == cluster_id
        profiles[int(cluster_id)] = {
            "name": segment_names.get(cluster_id, f"Segment {cluster_id}"),
            "count": int(mask.sum()),
            "pct": round(mask.sum() / len(labels) * 100, 2),
            "centroid_features": {f: round(float(centroids.loc[cluster_id, f]), 4) for f in feature_names},
        }

    return profiles


def compute_silhouette_details(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Per-sample silhouette with overall average."""
    sil_avg = silhouette_score(X_scaled, labels)
    sil_vals = silhouette_samples(X_scaled, labels)

    return {"silhouette_avg": round(sil_avg, 4), "per_sample": [round(float(v), 4) for v in sil_vals]}