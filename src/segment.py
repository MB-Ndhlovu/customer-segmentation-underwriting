import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_optimal_k(X_scaled, k_range: range, seed: int = 42):
    """Use Elbow method (inertia) and Silhouette score to pick best k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    # Pick k with best silhouette, bounded
    best_idx = int(np.argmax(silhouettes))
    best_k = list(k_range)[best_idx]
    best_silhouette = silhouettes[best_idx]

    return best_k, best_silhouette, inertias, silhouettes


def cluster(X_scaled, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(X_scaled)


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """Build per-segment profile summaries."""
    df_plot = df.copy()
    df_plot["cluster"] = labels
    profiles = df_plot.groupby("cluster")[feature_cols].mean()
    return profiles


def silhouette_detail(X_scaled, labels) -> dict:
    """Per-sample silhouette scores and overall stats."""
    sil_scores = silhouette_samples(X_scaled, labels)
    return {
        "mean": float(silhouette_score(X_scaled, labels)),
        "std": float(sil_scores.std()),
        "min": float(sil_scores.min()),
        "max": float(sil_scores.max()),
    }