"""KMeans clustering with Elbow + Silhouette analysis."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


def find_optimal_k(X: pd.DataFrame, k_range=range(2, 10)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    return inertias, silhouettes


def fit_kmeans(X: pd.DataFrame, n_clusters=4, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def silhouette_detail(X: pd.DataFrame, labels: np.ndarray):
    score = silhouette_score(X, labels)
    sample_scores = silhouette_samples(X, labels)
    return score, sample_scores


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df_out = df.copy()
    df_out["cluster"] = labels
    profiles = df_out.groupby("cluster").agg("mean").round(2)
    return profiles


def assign_segment_names(labels: np.ndarray, profiles_df: pd.DataFrame) -> dict:
    """Map cluster IDs to business segment names based on profiles."""
    segment_map = {}
    # Sort clusters by income to assign names consistently
    sorted_clusters = profiles_df.sort_values("income").index.tolist()
    name_map = {
        sorted_clusters[0]: "Mass Market",
        sorted_clusters[1]: "Rising Prime",
        sorted_clusters[2]: "Established Prime",
        sorted_clusters[3]: "Subprime High-Risk",
    }
    for cl in range(len(labels)):
        segment_map[cl] = name_map.get(cl, f"Cluster {cl}")
    return segment_map