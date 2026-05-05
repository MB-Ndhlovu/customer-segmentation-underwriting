import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X, k_range=(2, 10)):
    """Elbow method + silhouette analysis to pick best k."""
    inertias, silhouettes = [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    best_k = range(k_range[0], k_range[1] + 1)[np.argmax(silhouettes)]
    return best_k, inertias, silhouettes


def segment_customers(X, n_clusters=4):
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def profile_segments(df, labels):
    """Compute mean feature values per cluster."""
    df_copy = df.copy()
    df_copy["cluster"] = labels
    profiles = df_copy.groupby("cluster").mean(numeric_only=True)
    return profiles


def assign_segment_names(labels):
    """Map cluster IDs to business-friendly segment names by analyzing centroids."""
    label_counts = pd.Series(labels).value_counts().sort_index()
    return {i: f"Cluster {i} (n={label_counts.get(i,0)})" for i in range(labels.max() + 1)}