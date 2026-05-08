"""KMeans clustering with Elbow method, Silhouette analysis, and segment profiling."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)):
    """Compute inertia and silhouette scores for a range of k values."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  k={k}  inertia={inertia:.1f}  silhouette={sil:.4f}")
    return pd.DataFrame(results)


def plot_elbow(df_metrics: pd.DataFrame, output_path: str):
    """Save elbow plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_metrics["k"], df_metrics["inertia"], "bo-", linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved elbow plot: {output_path}")


def plot_silhouette(df_metrics: pd.DataFrame, output_path: str):
    """Save silhouette score plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_metrics["k"], df_metrics["silhouette"], "gs-", linewidth=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved silhouette plot: {output_path}")


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4):
    """Fit final KMeans model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def silhouette_analysis(X_scaled: np.ndarray, labels: np.ndarray, output_path: str):
    """Plot per-sample silhouette with annotations."""
    n_clusters = len(set(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(n_clusters):
        ith_silhouette_vals = silhouette_samples(X_scaled, labels)
        mask = labels == i
        size_i = mask.sum()
        vals = ith_silhouette_vals[mask][np.argsort(ith_silhouette_vals[mask])]
        y_upper = y_lower + size_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, vals,
            alpha=0.6
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    sil_avg = silhouette_score(X_scaled, labels)
    ax.axvline(sil_avg, color="red", linestyle="--", label=f"Avg={sil_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot (Avg={sil_avg:.3f})")
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved silhouette analysis: {output_path}")


def pca_visualization(X_scaled: np.ndarray, labels: np.ndarray, output_path: str):
    """2D PCA projection with cluster coloring."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.5, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA Projection of Customer Segments")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA plot: {output_path}")


def profile_segments(df: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute and display segment profiles."""
    X = X.copy()
    X["segment"] = labels

    counts = X["segment"].value_counts().sort_index()

    print("\n" + "=" * 70)
    print("SEGMENT PROFILES")
    print("=" * 70)
    for seg in sorted(counts.index):
        name = SEGMENT_NAMES.get(seg, f"Segment {seg}")
        count = counts[seg]
        pct = count / len(labels) * 100
        print(f"\nSegment {seg}: {name}  (n={count}, {pct:.1f}%)")

    print("\n--- Feature Means by Segment ---")
    means = X.groupby("segment").mean()
    print(means.round(4).to_string())

    profile = X.groupby("segment").agg(["mean", "std", "min", "max"])
    return profile


def save_results(df_metrics: pd.DataFrame, profile: pd.DataFrame,
                km: KMeans, labels: np.ndarray, sil: float,
                report_path: str):
    """Save segmentation results to JSON."""
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}

    results = {
        "optimal_k": 4,
        "k_tested_range": list(range(2, 10)),
        "metrics": df_metrics.to_dict(orient="records"),
        "silhouette_avg": float(sil),
        "segment_counts": counts,
        "segment_names": SEGMENT_NAMES,
        "inertia": float(km.inertia_),
    }

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {report_path}")

    return results


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, scale_features

    df = generate_customer_data()
    X = build_features(df)
    X_scaled, scaler = scale_features(X)

    print("Finding optimal k...")
    df_metrics = find_optimal_k(X_scaled)
    print(df_metrics)

    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"\nFinal k=4 silhouette: {sil:.4f}")

    profile = profile_segments(df, X, labels)