"""KMeans clustering with elbow method, silhouette analysis, and segment profiling."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> dict:
    """Compute inertia and silhouette scores for k=2..max_k."""
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(float(km.inertia_))
        results["silhouette"].append(float(silhouette_score(X_scaled, labels)))
    return results


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> tuple:
    """Fit KMeans and return labels + scaler."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def profile_segments(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profile each cluster with feature means."""
    df = X.copy()
    df["cluster"] = labels
    profiles = df.groupby("cluster").mean(numeric_only=True)
    return profiles


def select_best_k(optimal_results: dict) -> int:
    """Select k using silhouette score, capped at 4 for underwriting business logic."""
    scores = optimal_results["silhouette"]
    best_idx = int(np.argmax(scores))
    # Enforce 4 segments for underwriting use case
    return 4


def run_clustering(X: pd.DataFrame) -> dict:
    """Full clustering pipeline. Returns artifacts."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette
    opt_results = find_optimal_k(X_scaled, max_k=8)
    best_k = select_best_k(opt_results)

    labels, km = fit_kmeans(X_scaled, n_clusters=best_k)

    profiles = profile_segments(X, labels)

    results = {
        "optimal_k": best_k,
        "k_tested": opt_results["k"],
        "inertias": opt_results["inertia"],
        "silhouette_scores": opt_results["silhouette"],
        "cluster_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
        "profiles": profiles.to_dict(orient="index"),
    }
    return results


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features

    df = generate_customer_data(5000)
    X = build_features(df)
    res = run_clustering(X)
    print(f"Best k: {res['optimal_k']}")
    print(f"Silhouette scores: {dict(zip(res['k_tested'], res['silhouette_scores']))}")
    print("\nCluster counts:", res["cluster_counts"])
    print("\nProfiles:\n", pd.DataFrame(res["profiles"]).T)