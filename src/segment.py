"""KMeans clustering with Elbow method, Silhouette analysis, and segment profiling."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Run Elbow method + Silhouette analysis to find optimal k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Best k = highest silhouette score
    best_k = list(k_range)[np.argmax(silhouettes)]

    return {
        "k_values": list(k_range),
        "inertias": [round(v, 2) for v in inertias],
        "silhouettes": [round(v, 4) for v in silhouettes],
        "best_k": best_k,
    }


def cluster_customers(X_scaled, n_clusters=4, seed=42) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def map_clusters_to_segments(df, labels):
    """Map raw cluster IDs to meaningful segment names based on profile characteristics."""
    df = df.copy()
    df["segment_label"] = labels

    # Compute cluster profiles
    cluster_profiles = df.groupby("segment_label")[[
        "income", "credit_score", "employment_years", "debt_to_income", "loan_history_count", "age"
    ]].mean()

    # Score each cluster on underwriting quality
    def score_cluster(row):
        score = 0
        score += row["income"] / 100000
        score += row["credit_score"] / 1000
        score += row["employment_years"] / 15
        score -= row["debt_to_income"] * 3
        score -= row["loan_history_count"] / 5
        return score

    cluster_profiles["quality_score"] = cluster_profiles.apply(score_cluster, axis=1)
    cluster_profiles = cluster_profiles.sort_values("quality_score")

    # Map quality ordering to segment names
    # Lowest quality = Subprime High-Risk, Highest = Established Prime
    sorted_clusters = cluster_profiles.index.tolist()
    mapping = {
        sorted_clusters[0]: 3,  # Subprime High-Risk
        sorted_clusters[1]: 0,   # Mass Market
        sorted_clusters[2]: 1,   # Rising Prime
        sorted_clusters[3]: 2,    # Established Prime
    }

    df["segment_label"] = df["segment_label"].map(mapping)
    return df


def profile_segments(df: pd.DataFrame) -> dict:
    """Generate detailed segment profiles."""
    profiles = {}
    for seg_id in sorted(df["segment_label"].unique()):
        seg_df = df[df["segment_label"] == seg_id]
        profiles[SEGMENT_NAMES[seg_id]] = {
            "count": int(len(seg_df)),
            "pct": round(len(seg_df) / len(df) * 100, 1),
            "income_mean": round(seg_df["income"].mean(), 0),
            "credit_score_mean": round(seg_df["credit_score"].mean(), 1),
            "employment_years_mean": round(seg_df["employment_years"].mean(), 2),
            "debt_to_income_mean": round(seg_df["debt_to_income"].mean(), 3),
            "loan_history_count_mean": round(seg_df["loan_history_count"].mean(), 2),
            "age_mean": round(seg_df["age"].mean(), 1),
            "home_ownership_rate": round(seg_df["home_ownership"].mean(), 3),
            "verified_income_rate": round(seg_df["verified_income"].mean(), 3),
        }
    return profiles


def run_segmentation(X_scaled, df, n_clusters=4):
    """Run full segmentation pipeline."""
    # Find optimal k
    opt = find_optimal_k(X_scaled)

    # Use specified k (default 4)
    labels, km = cluster_customers(X_scaled, n_clusters=n_clusters)

    # Map clusters to meaningful segments
    df_labeled = map_clusters_to_segments(df, labels)

    # Profile segments
    profiles = profile_segments(df_labeled)

    sil_score = round(silhouette_score(X_scaled, df_labeled["segment_label"]), 4)

    results = {
        "optimal_k_analysis": opt,
        "silhouette_score": sil_score,
        "segment_profiles": profiles,
        "segment_names": SEGMENT_NAMES,
        "n_clusters": n_clusters,
    }

    return df_labeled, results, km


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, scale_features

    df = generate_customer_data()
    df_feat = build_features(df)
    X_scaled, scaler = scale_features(df_feat)
    df_labeled, results, km = run_segmentation(X_scaled, df_feat)
    print(json.dumps(results, indent=2))