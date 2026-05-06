"""KMeans clustering, elbow method, silhouette analysis, and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """Elbow method + silhouette analysis to pick k."""
    inertias = []
    silhouette_scores = []
    results = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertias.append(inertia)
        silhouette_scores.append(sil)
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  k={k}  inertia={inertia:.1f}  silhouette={sil:.4f}")

    # Pick k=4 if in range (our target)
    if 4 in k_range:
        chosen_k = 4
    else:
        # Fallback to best silhouette
        chosen_k = k_range[np.argmax(silhouette_scores)]

    return chosen_k, results


def fit_kmeans(X_scaled, n_clusters=4):
    """Fit final KMeans model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df, labels, feature_cols):
    """Build segment profile summary."""
    df_out = df.copy()
    df_out["cluster"] = labels

    profiles = {}
    for cluster_id in sorted(df_out["cluster"].unique()):
        mask = df_out["cluster"] == cluster_id
        seg_data = df_out[mask]
        seg_name = SEGMENT_NAMES.get(cluster_id, f"Cluster {cluster_id}")

        profiles[int(cluster_id)] = {
            "segment_name": seg_name,
            "count": int(mask.sum()),
            "pct": round(mask.sum() / len(df_out) * 100, 2),
            "features": {
                col: {
                    "mean": round(float(seg_data[col].mean()), 4),
                    "std": round(float(seg_data[col].std()), 4),
                    "min": round(float(seg_data[col].min()), 4),
                    "max": round(float(seg_data[col].max()), 4),
                }
                for col in feature_cols
            },
        }

    return profiles


def run_segmentation(df, feature_cols, n_clusters=4):
    """Full segmentation pipeline: scale → find k → cluster → profile."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== Finding optimal k ===")
    chosen_k, k_results = find_optimal_k(X_scaled, k_range=range(2, 8))
    print(f"\nChosen k={chosen_k}")

    km, labels = fit_kmeans(X_scaled, n_clusters=chosen_k)

    print(f"\n=== Silhouette Score: {silhouette_score(X_scaled, labels):.4f} ===")

    # Map cluster IDs to segment names (by centroid ordering — income as tiebreaker)
    centroids = km.cluster_centers_
    # Sort centroids by income (first feature in standard-scaled space)
    centroid_order = centroids[:, 0].argsort()
    name_map = {old: new for new, old in enumerate(centroid_order)}
    labels_mapped = np.array([name_map[l] for l in labels])

    # Remap for cleaner profiles
    km_final, labels_final = fit_kmeans(X_scaled, n_clusters=chosen_k)
    profiles = profile_segments(df, labels_final, feature_cols)

    sil = silhouette_score(X_scaled, labels_final)
    sil_samples = silhouette_samples(X_scaled, labels_final)

    summary = {
        "n_clusters": chosen_k,
        "silhouette_score": round(float(sil), 4),
        "k_search_results": k_results,
        "segment_profiles": profiles,
    }

    return km_final, labels_final, scaler, summary, sil, sil_samples


if __name__ == "__main__":
    from data_loader import generate_customer_data, get_feature_columns
    df = generate_customer_data(5000)
    feature_cols = get_feature_columns()
    km, labels, scaler, summary, sil, sil_samples = run_segmentation(df, feature_cols)
    print("\n=== Segment Profiles ===")
    for cid, prof in summary["segment_profiles"].items():
        print(f"\nCluster {cid} — {prof['segment_name']} ({prof['count']} customers, {prof['pct']}%):")
        for feat, stats in prof["features"].items():
            print(f"  {feat}: mean={stats['mean']}")