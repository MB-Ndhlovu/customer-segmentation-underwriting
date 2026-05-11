import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def elbow_method(X: np.ndarray, k_range: range, random_state: int = 42):
    """Compute inertia and distortion for elbow analysis."""
    inertias, distortions = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        # distortion = mean squared distance to closest centroid
        distortions.append(
            np.mean(np.min(cdist(X, km.cluster_centers_, "euclidean"), axis=1))
        )
    return inertias, distortions


def find_best_k(X: np.ndarray, k_range: range, random_state: int = 42):
    """Select k with highest average silhouette score."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """Fit KMeans with specified k; returns model, labels, and metrics."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    sil_samples = silhouette_samples(X, labels)
    return model, labels, sil_score, sil_samples


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build per-segment profile summary from original features."""
    df = df.copy()
    df["segment_label"] = labels
    profiles = df.groupby("segment_label").agg({
        "income": ["mean", "std"],
        "credit_score": ["mean", "std"],
        "employment_years": ["mean", "std"],
        "debt_to_income": ["mean", "std"],
        "loan_history_count": ["mean", "std"],
        "age": ["mean", "std"],
        "home_ownership": "mean",
        "verified_income": "mean",
    }).round(2)
    return profiles


def assign_segment_names(labels: np.ndarray) -> np.ndarray:
    """Map KMeans cluster IDs to business-relevant names via heuristics."""
    # Use centroids to determine which cluster corresponds to which segment
    # We'll identify by highest credit_score + income cluster = Established Prime etc.
    return labels  # Names mapped in reporting
