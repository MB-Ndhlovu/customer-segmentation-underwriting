"""KMeans clustering with Elbow method and Silhouette analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Tuple, List


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
) -> Tuple[List[float], List[float]]:
    """Compute inertia and silhouette scores for a range of K."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return inertias, silhouettes


def plot_elbow_silhouette(
    k_range: range,
    inertias: List[float],
    silhouettes: List[float],
    output_path: str = "reports/elbow_silhouette.png",
) -> None:
    """Save Elbow + Silhouette combo plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True)

    axes[1].plot(list(k_range), silhouettes, "go-", linewidth=2)
    axes[1].axvline(x=4, color="r", linestyle="--", label="Selected K=4")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> KMeans:
    """Fit KMeans with specified K."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


def get_cluster_profiles(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute mean feature values per cluster."""
    df_labels = pd.DataFrame(labels, columns=["cluster"])
    df_labels["cluster"] = df_labels["cluster"].map(SEGMENT_NAMES)
    df_profiles = X.copy()
    df_profiles["cluster"] = df_labels["cluster"].values

    profile = df_profiles.groupby("cluster")[feature_cols].mean().round(2)

    # Reorder to logical risk sequence
    order = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
    profile = profile.reindex([o for o in order if o in profile.index])

    return profile


def run_segmentation(
    X_scaled: np.ndarray,
    X_orig: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 4,
    output_dir: str = "reports",
) -> Tuple[KMeans, np.ndarray, dict]:
    """Run full segmentation: optimal K analysis, clustering, profiling."""
    k_range = range(2, 11)
    inertias, silhouettes = find_optimal_k(X_scaled, k_range)

    plot_elbow_silhouette(
        k_range=k_range,
        inertias=inertias,
        silhouettes=silhouettes,
        output_path=f"{output_dir}/elbow_silhouette.png",
    )

    km = fit_kmeans(X_scaled, n_clusters=n_clusters)
    labels = km.labels_

    silhouette_avg = silhouette_score(X_scaled, labels)

    profile = get_cluster_profiles(X_orig, labels, feature_cols)

    sample_silhouette = silhouette_samples(X_scaled, labels)

    results = {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "silhouette_avg": silhouette_avg,
        "cluster_counts": dict(pd.Series(labels).value_counts().sort_index()),
        "profile": profile.to_dict(),
        "selected_k": n_clusters,
    }

    return km, labels, results