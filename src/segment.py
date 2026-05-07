"""Unsupervised segmentation via KMeans — Elbow + Silhouette analysis."""

import json
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> Tuple[int, dict]:
    """Evaluate K across range; return optimal k and metrics dict."""
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Pick k with highest silhouette, but minimum 4 for business requirement
    optimal_k = max(k_range, key=lambda k: silhouettes[k - k_range.start])
    optimal_k = max(optimal_k, 4)  # enforce minimum 4

    metrics = {
        "k_values": list(k_range),
        "inertias": [float(i) for i in inertias],
        "silhouettes": [float(s) for s in silhouettes],
        "optimal_k": optimal_k,
    }
    return optimal_k, metrics


def cluster(X_scaled: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X_scaled)


def profile_segments(X: pd.DataFrame, labels: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean feature values per cluster label."""
    X = X.copy()
    X["cluster"] = labels
    # Add raw cols for profiling
    for col in ["income", "credit_score", "employment_years",
                "debt_to_income", "loan_history_count", "age",
                "home_ownership", "verified_income"]:
        X[col] = df[col].values

    profiles = X.groupby("cluster").mean(numeric_only=True).round(3)
    return profiles


def save_results(profiles: pd.DataFrame, metrics: dict, path: str):
    result = {
        "optimal_k": metrics["optimal_k"],
        "silhouettes_by_k": dict(zip(metrics["k_values"], metrics["silhouettes"])),
        "cluster_profiles": profiles.to_dict(orient="index"),
        "business_segment_mapping": {
            "0": "Mass Market",
            "1": "Rising Prime",
            "2": "Established Prime",
            "3": "Subprime High-Risk",
        },
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {path}")