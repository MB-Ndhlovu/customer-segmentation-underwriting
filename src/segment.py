"""KMeans clustering with Elbow and Silhouette analysis for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

LABEL_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def find_optimal_k(X_scaled, k_range=(2, 8)):
    """Run Elbow + Silhouette analysis. Returns (inertias, silhouette_scores)."""
    inertias = []
    silhouette_scores = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
        else:
            silhouette_scores.append(0.0)
    return inertias, silhouette_scores


def fit_kmeans(X_scaled, k=4):
    """Fit KMeans with k=4, label-aligned to business segments by centroid ordering."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Reorder labels by centroid distance from ideal anchor points
    # Anchor: highest income+credit for Prime end, lowest for subprime
    centroids = km.cluster_centers_
    # Sort centroids by income (col 0) then credit_score (col 1) descending
    order = np.argsort(centroids[:, 0] + centroids[:, 1] * 10)[::-1]
    centroid_map = {old: new for new, old in enumerate(order)}

    # Map cluster indices to label names
    label_names = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
    return km, labels, centroid_map, label_names


def assign_label_names(labels, centroid_map):
    return np.array([centroid_map[l] for l in labels])


def profile_segments(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute mean feature values per segment."""
    X = X.copy()
    X["segment"] = labels
    profiles = X.groupby("segment").mean().round(2)
    profiles["count"] = X.groupby("segment").size()
    return profiles


if __name__ == "__main__":
    from data_loader import load_customer_data
    from features import build_features

    df = load_customer_data()
    X = build_features(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    inertias, silhouettes = find_optimal_k(Xs, (2, 7))
    print("Inertias:", inertias)
    print("Silhouettes:", silhouettes)

    km, labels, cmap, _ = fit_kmeans(Xs, k=4)
    profile = profile_segments(X, labels)
    print("\nSegment profiles:")
    print(profile)