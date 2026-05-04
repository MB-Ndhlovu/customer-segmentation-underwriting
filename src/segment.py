"""KMeans clustering pipeline: Elbow method, Silhouette analysis, segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import json


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)):
    """Find optimal k using Elbow (inertia) and Silhouette scores."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[np.argmax(silhouettes)]
    return best_k, inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, seed: int = 42):
    """Fit KMeans with specified k."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build a profile table for each cluster."""
    df = df.copy()
    df["cluster"] = labels

    numeric_cols = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age",
    ]

    profile = df.groupby("cluster")[numeric_cols].mean().round(2)
    profile["count"] = df.groupby("cluster").size()

    # Assign business-friendly segment names by ordering clusters by income
    income_order = profile["income"].sort_values(ascending=False).index.tolist()
    name_map = {
        income_order[0]: "Established Prime",
        income_order[1]: "Rising Prime",
        income_order[2]: "Mass Market",
        income_order[3]: "Subprime High-Risk",
    }
    profile["segment_name"] = profile.index.map(name_map)

    return profile


def compute_silhouette_details(X_scaled: np.ndarray, labels: np.ndarray):
    """Compute per-sample silhouette scores and overall mean."""
    sample_scores = silhouette_samples(X_scaled, labels)
    mean_score = silhouette_score(X_scaled, labels)
    return sample_scores, mean_score


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features for clustering."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return X, scaler


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4):
    """Main segmentation function.

    Returns (labels, profile_df, silhouette_mean, scaler, km_model)
    """
    from src.features import build_features, get_clustering_features

    df_feat = build_features(df)
    feature_df = get_clustering_features(df_feat)

    X_scaled, scaler = scale_features(feature_df)

    # Run Elbow + Silhouette analysis
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)

    # Fit final model
    km, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    sample_scores, silhouette_mean = compute_silhouette_details(X_scaled, labels)

    # Profile segments
    df_with_clusters = df_feat.copy()
    df_with_clusters["cluster"] = labels
    profile = profile_segments(df_with_clusters, labels)

    return {
        "labels": labels,
        "profile": profile,
        "silhouette_mean": round(silhouette_mean, 4),
        "inertias": [round(i, 2) for i in inertias],
        "silhouettes": [round(s, 4) for s in silhouettes],
        "optimal_k": best_k,
        "scaler": scaler,
        "km_model": km,
    }


if __name__ == "__main__":
    from src.data_loader import generate_customer_data

    df = generate_customer_data()
    result = run_segmentation(df)
    print("Silhouette Score:", result["silhouette_mean"])
    print(result["profile"])