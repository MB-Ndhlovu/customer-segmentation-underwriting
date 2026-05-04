import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

# Feature columns used for clustering
FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership",
    "verified_income",
]


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Evaluate KMeans for k in k_range using silhouette score and inertia.
    Returns dict with 'k', 'silhouette', 'inertia', 'elbow'.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette": sil, "inertia": km.inertia_})

    # Elbow: find k where inertia drop starts flattening (second derivative)
    inertias = [r["inertia"] for r in results]
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = int(np.argmax(second_diffs) + 2)  # +2 to account for double diff offset
    elbow_k = results[min(elbow_idx, len(results) - 1)]["k"]

    best = max(results, key=lambda x: x["silhouette"])

    return {
        "k_optimal": int(best["k"]),
        "k_elbow": int(elbow_k),
        "best_silhouette": float(best["silhouette"]),
        "all_results": results,
    }


def assign_segment_names(labels: np.ndarray) -> pd.Series:
    """Map cluster IDs to business segment names based on centroid analysis."""
    return pd.Series(labels).map(SEGMENT_NAMES)


def profile_segment(segment_df: pd.DataFrame, segment_name: str) -> dict:
    """Generate a statistical profile for a segment."""
    numeric_cols = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
    ]
    profile = {}
    for col in numeric_cols:
        profile[col] = {
            "mean": float(segment_df[col].mean()),
            "std": float(segment_df[col].std()),
            "median": float(segment_df[col].median()),
            "min": float(segment_df[col].min()),
            "max": float(segment_df[col].max()),
        }
    profile["home_ownership_rate"] = float(segment_df["home_ownership"].mean())
    profile["verified_income_rate"] = float(segment_df["verified_income"].mean())
    profile["n_customers"] = int(len(segment_df))
    return profile


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Full segmentation pipeline: scale features, run KMeans, profile clusters.
    Returns df with 'segment_label' and 'segment_name' columns.
    """
    # Select base features
    X = df[FEATURE_COLS].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k
    opt = find_optimal_k(X_scaled)
    print(f"[segment] Optimal k (silhouette): {opt['k_optimal']}  |  Elbow k: {opt['k_elbow']}")

    # Fit final KMeans — use requested n_clusters (business requirement = 4)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df["segment_label"] = labels

    # Determine which cluster ID corresponds to which business segment
    # We map based on centroid characteristics: credit_score and income are most indicative
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=FEATURE_COLS,
    )
    centroids["cluster"] = range(n_clusters)

    # Rank clusters by composite score: high credit + high income = Prime
    centroids["prime_score"] = (
        centroids["credit_score"] / 850 + centroids["income"] / 120000
    ) / 2

    # Rank clusters by risk: high DTI + high loans = High-Risk
    centroids["risk_score"] = (
        centroids["debt_to_income"] / 0.6 + centroids["loan_history_count"] / 6
    ) / 2

    # Sort clusters into business segments
    sorted_by_prime = centroids.sort_values("prime_score", ascending=False)["cluster"].tolist()
    sorted_by_risk = centroids.sort_values("risk_score", ascending=False)["cluster"].tolist()

    # Map cluster IDs to segment labels (0=best, 3=worst)
    # Established Prime = highest prime_score, lowest risk
    # Subprime High-Risk = highest risk_score
    # Rising Prime = second highest prime_score
    # Mass Market = middle ground
    cluster_to_label = {}
    cluster_to_label[sorted_by_prime[0]] = 2  # Established Prime
    cluster_to_label[sorted_by_prime[1]] = 1  # Rising Prime
    cluster_to_label[sorted_by_prime[2]] = 0  # Mass Market
    cluster_to_label[sorted_by_risk[0]] = 3  # Subprime High-Risk (only if not already assigned)

    # Ensure all clusters are assigned
    for c in range(n_clusters):
        if c not in cluster_to_label:
            cluster_to_label[c] = 0  # fallback

    df["segment_label"] = df["segment_label"].map(cluster_to_label)
    df["segment_name"] = df["segment_label"].map(SEGMENT_NAMES)

    # Profiling
    profiles = {}
    for label, name in SEGMENT_NAMES.items():
        seg_df = df[df["segment_label"] == label]
        profiles[name] = profile_segment(seg_df, name)

    silhouette = float(silhouette_score(X_scaled, df["segment_label"].values))

    results = {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "optimal_k_analysis": opt,
        "centroids": centroids.to_dict(orient="records"),
        "profiles": profiles,
        "segment_counts": df["segment_label"].value_counts().sort_index().to_dict(),
    }

    return df, results


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_features

    df = generate_synthetic_data()
    df_feat = build_features(df)
    df_seg, results = run_segmentation(df_feat)
    print(results["segment_counts"])
    print(json.dumps(results["profiles"], indent=2))