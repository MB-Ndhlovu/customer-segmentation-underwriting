"""KMeans clustering with Elbow method and Silhouette analysis."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)):
    """Run Elbow + Silhouette analysis to pick best k."""
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[np.argmax(silhouettes)]
    return best_k, inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4):
    """Fit KMeans with 4 clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute mean feature values per segment."""
    df_out = df.copy()
    df_out["segment_label"] = labels

    profile = df_out.groupby("segment_label").mean(numeric_only=True).round(3)
    profile["count"] = df_out.groupby("segment_label").size()
    profile["segment_name"] = profile.index.map(SEGMENT_NAMES)
    profile = profile[["segment_name", "count", "income", "credit_score",
                       "employment_years", "debt_to_income", "loan_history_count",
                       "age", "home_ownership", "verified_income"]]
    return profile


def run_segmentation(X_scaled: np.ndarray, feature_names: list):
    """Full segmentation pipeline: find k, fit, profile."""
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)

    # Force 4 for business requirement
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil_score = silhouette_score(X_scaled, labels)

    return {
        "best_k_observed": int(best_k),
        "k_used": 4,
        "silhouette_score": round(float(sil_score), 4),
        "inertias": [round(float(i), 2) for i in inertias],
        "silhouettes": [round(float(s), 4) for s in silhouettes],
    }, labels, km


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_names

    df = generate_customer_data(5000)
    X = build_features(df)
    feature_names = get_feature_names()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_names])

    stats, labels, km = run_segmentation(X_scaled, feature_names)
    print("Segmentation stats:", stats)
    profile = profile_segments(X, labels)
    print("\nSegment profiles:\n", profile)