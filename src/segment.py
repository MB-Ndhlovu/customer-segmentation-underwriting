"""KMeans clustering with Elbow method, Silhouette analysis, and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import json


def find_optimal_k(X_scaled: np.ndarray, k_range: range, random_state: int = 42):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(np.nan)

    return inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, k: int = 4, random_state: int = 42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def silhouette_details(X_scaled: np.ndarray, labels: np.ndarray):
    sil_avg = silhouette_score(X_scaled, labels)
    sil_samples = silhouette_samples(X_scaled, labels)
    return sil_avg, sil_samples


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> dict:
    df = df.copy()
    df["cluster"] = labels

    profiles = {}
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        profile = {}
        for col in feature_cols:
            profile[col] = {
                "mean": round(float(sub[col].mean()), 4),
                "std": round(float(sub[col].std()), 4),
                "min": round(float(sub[col].min()), 4),
                "max": round(float(sub[col].max()), 4),
            }
        profile["count"] = int(len(sub))
        profiles[int(cluster_id)] = profile

    return profiles


def label_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """Map numeric cluster IDs to human-readable segment names based on profiling."""
    df = df.copy()
    df["cluster"] = labels

    # Build a simple rule-based labeler based on known centroids
    # We use the profile to determine which cluster maps to which segment
    cluster_income = df.groupby("cluster")["income"].mean()
    cluster_credit = df.groupby("cluster")["credit_score"].mean()
    cluster_dti = df.groupby("cluster")["debt_to_income"].mean()

    # Sort clusters by income descending to assign names
    sorted_clusters = cluster_income.sort_values(ascending=False).index.tolist()

    name_map = {
        sorted_clusters[0]: "Established Prime",
        sorted_clusters[1]: "Rising Prime",
        sorted_clusters[2]: "Mass Market",
        sorted_clusters[3]: "Subprime High-Risk",
    }

    # Check the DTI to correctly identify Subprime
    for cluster_id in sorted_clusters:
        if cluster_dti[cluster_id] > 0.30:
            name_map[cluster_id] = "Subprime High-Risk"

    df["segment_name"] = df["cluster"].map(name_map)
    return df


def run_clustering(df: pd.DataFrame, feature_cols: list, k: int = 4, random_state: int = 42):
    # Scale features
    scaler = StandardScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette
    inertias, silhouettes = find_optimal_k(X_scaled, range(2, 9), random_state)

    # Fit final model
    km, labels = fit_kmeans(X_scaled, k=k, random_state=random_state)
    sil_avg, sil_samples = silhouette_details(X_scaled, labels)

    # Profile
    profiles = profile_segments(df, labels, feature_cols)

    # Label
    df_labeled = label_segments(df, labels, feature_cols)

    results = {
        "k": k,
        "inertias": [round(float(i), 4) for i in inertias],
        "silhouette_scores": [round(float(s), 4) if not np.isnan(s) else None for s in silhouettes],
        "silhouette_avg": round(float(sil_avg), 4),
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in sorted(set(labels))},
        "profiles": profiles,
    }

    return df_labeled, labels, scaler, km, results