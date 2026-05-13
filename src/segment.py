"""KMeans clustering: elbow method, silhouette analysis, segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


SEGMENT_LABELS = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> Tuple[int, dict]:
    """Elbow method + silhouette scoring to pick best k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Elbow: find kink using rate-of-change
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = k_range.start + int(np.argmax(diffs2)) + 1

    best_k = max(2, elbow_k)
    best_sil = max(silhouettes)

    # Fallback silhouette-based choice if elbow is off
    sil_k = k_range.start + int(np.argmax(silhouettes))
    if sil_k > best_k:
        best_k = sil_k

    return best_k, {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> Tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def assign_segment_names(labels: np.ndarray) -> pd.Series:
    # We must remap clusters to match business profile ordering
    # Cluster centroids in scaled space — find which cluster corresponds to which segment
    # Profiles: 0=mid, 1=good, 2=best, 3=worst
    return pd.Series(labels).map(lambda x: SEGMENT_LABELS[x])


def profile_segments(
    df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]
) -> pd.DataFrame:
    """Print and return mean feature values per cluster."""
    df_out = df.copy()
    df_out["cluster"] = labels
    profile = df_out.groupby("cluster")[feature_cols].mean()
    return profile


if __name__ == "__main__":
    from .data_loader import load_customer_data
    from .features import build_features, get_feature_columns

    df = load_customer_data()
    df = build_features(df)
    feat_cols = get_feature_columns()

    X = df[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k, diag = find_optimal_k(X_scaled, range(2, 11))
    print(f"Optimal k = {k}, silhouettes = {diag['silhouettes']}")

    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil:.4f}")

    profile = profile_segments(df, labels, feat_cols)
    print(profile)