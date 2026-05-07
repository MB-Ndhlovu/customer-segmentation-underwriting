"""KMeans clustering with Elbow + Silhouette analysis and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}


def find_optimal_k(X_scaled: pd.DataFrame, k_range: range = None) -> dict:
    """
    Run Elbow method (inertia) and Silhouette analysis for k in k_range.
    Returns dict with inertia, silhouette scores per k.
    """
    if k_range is None:
        k_range = range(2, 10)

    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        results[k] = {'inertia': inertia, 'silhouette': sil}

    return results


def fit_kmeans(X_scaled: pd.DataFrame, n_clusters: int = 4, **kwargs) -> tuple:
    """Fit KMeans and return (labels, model)."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
    labels = km.fit_predict(X_scaled)
    return labels, km


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Create a profile table (mean feature values per cluster) and
    map cluster IDs → segment names based on financial logic.
    """
    df_labeled = df.copy()
    df_labeled['cluster'] = labels

    profile = df_labeled.groupby('cluster').mean(numeric_only=True)

    # Name clusters based on characteristic patterns
    # We'll map by looking at income + credit_score rank
    cluster_scores = {}
    for c in profile.index:
        cluster_scores[c] = profile.loc[c, 'income'] + profile.loc[c, 'credit_score'] * 500

    rank_order = sorted(cluster_scores, key=cluster_scores.get)
    cluster_to_name = {rank_order[i]: SEGMENT_NAMES[i] for i in range(len(rank_order))}

    profile['segment_name'] = profile.index.map(cluster_to_name)
    profile['segment_label'] = profile.index.map(
        {v: k for k, v in cluster_to_name.items()}
    )

    return profile


def run_clustering(X_scaled: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Full clustering workflow:
      - find_optimal_k
      - fit KMeans
      - profile segments
    Returns dict with labels, profile, and metrics.
    """
    labels, km = fit_kmeans(X_scaled, n_clusters)
    sil = silhouette_score(X_scaled, labels)

    return {
        'labels': labels,
        'model': km,
        'silhouette': sil,
        'n_clusters': n_clusters,
    }


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features, scale_features, get_feature_names

    df = generate_customer_data(5000)
    X = build_features(df)
    X_scaled, scaler = scale_features(X)

    opt = find_optimal_k(X_scaled)
    print("K | Inertia | Silhouette")
    for k, v in opt.items():
        print(f"{k}: {v['inertia']:.1f} | {v['silhouette']:.4f}")

    result = run_clustering(X_scaled, n_clusters=4)
    print(f"\nSilhouette score: {result['silhouette']:.4f}")