"""
Segmentation pipeline: KMeans clustering with Elbow + Silhouette analysis.
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run_elbow_silhouette(X_scaled: np.ndarray, max_k: int = 10):
    """
    Compute inertia and silhouette scores for k=1..max_k.
    Returns dicts of k -> inertia and k -> silhouette.
    """
    inertias = {}
    silhouettes = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(silhouette_score(X_scaled, km.labels_))
    return inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4):
    """Fit KMeans with specified k, return model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> dict:
    """
    Compute mean (and std) per cluster across all features.
    Maps cluster ID -> segment name based on characteristic ordering.
    Returns profiling dict + ordered list of segment names.
    """
    df_plot = df.copy()
    df_plot["_cluster"] = labels

    profiles = {}
    segment_order = []

    for cluster_id in sorted(df_plot["_cluster"].unique()):
        sub = df_plot[df_plot["_cluster"] == cluster_id]
        row = {}
        for col in feature_cols:
            row[col] = {
                "mean": float(sub[col].mean()),
                "std": float(sub[col].std()),
            }
        profiles[int(cluster_id)] = row

    # Identify each cluster by its most distinctive characteristic
    # Strategy: rank by credit_score and income
    cluster_stats = []
    for cid, prof in profiles.items():
        cs = prof["credit_score"]["mean"]
        inc = prof["income"]["mean"]
        dti = prof["debt_to_income"]["mean"]
        cluster_stats.append((cid, cs, inc, dti))

    # Sort by credit_score then income to assign labels
    cluster_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)

    SEGMENT_LABELS = [
        "Established Prime",
        "Rising Prime",
        "Mass Market",
        "Subprime High-Risk",
    ]

    mapping = {}
    for label, (cid, *_rest) in zip(SEGMENT_LABELS, cluster_stats):
        mapping[int(cid)] = label
        segment_order.append(label)

    return mapping, segment_order, profiles


def run_segmentation(df: pd.DataFrame, feature_cols: list, n_clusters: int = 4):
    """
    Full segmentation pipeline.
    Returns: labels, scaler, kmeans_model, segment_mapping, profiles
    """
    scaler = StandardScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette
    inertias, silhouettes = run_elbow_silhouette(X_scaled, max_k=10)

    # Fit final model
    km = fit_kmeans(X_scaled, n_clusters=n_clusters)
    labels = km.labels_

    # Profile
    mapping, segment_order, profiles = profile_segments(df, labels, feature_cols)

    return {
        "labels": labels,
        "scaler": scaler,
        "model": km,
        "mapping": mapping,
        "segment_order": segment_order,
        "profiles": profiles,
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def save_results(results: dict, output_path: str):
    """Serialize segmentation results to JSON."""
    payload = {
        "segment_mapping": results["mapping"],
        "segment_order": results["segment_order"],
        "profiles": results["profiles"],
        "inertias": results["inertias"],
        "silhouettes": results["silhouettes"],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path