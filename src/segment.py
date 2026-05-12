"""KMeans clustering with elbow method and silhouette analysis."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X: np.ndarray, k_range: range, random_state: int = 42) -> dict:
    """Run elbow method + silhouette analysis across k_range."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbls = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, lbls)))
    return {"k_values": list(k_range), "inertias": inertias, "silhouettes": silhouettes}


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> dict:
    """Compute mean feature values per segment."""
    df_temp = df.copy()
    df_temp["cluster"] = labels
    profiles = {}
    for cid in sorted(df_temp["cluster"].unique()):
        seg_df = df_temp[df_temp["cluster"] == cid]
        profiles[int(cid)] = {
            "name": SEGMENT_NAMES.get(int(cid), f"Segment {cid}"),
            "count": int(len(seg_df)),
            "pct": round(len(seg_df) / len(df_temp) * 100, 2),
            "means": {col: round(float(seg_df[col].mean()), 4) for col in feature_cols},
        }
    return profiles


def run_clustering(df: pd.DataFrame, feature_cols: list, n_clusters: int = 4) -> dict:
    """Main clustering routine."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette sweep
    analysis = find_optimal_k(X_scaled, range(2, 9))

    # Final KMeans model
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil_score = float(silhouette_score(X_scaled, labels))
    profiles = profile_segments(df, labels, feature_cols)

    # Map clusters → business segment labels using profile analysis
    named_label_idx = _assign_segment_names(df, labels, feature_cols)

    return {
        "labels": labels,
        "named_labels": named_label_idx,
        "inertias": analysis["inertias"],
        "silhouettes": analysis["silhouettes"],
        "k_values": analysis["k_values"],
        "silhouette_score": round(sil_score, 4),
        "profiles": profiles,
        "scaler": scaler,
        "km": km,
    }


def _assign_segment_names(df, labels, feature_cols) -> np.ndarray:
    """Score each cluster on prime/risk dimensions, then map to segment names."""
    df_temp = df.copy()
    df_temp["cluster"] = labels
    cluster_means = df_temp.groupby("cluster")[feature_cols].mean()

    # Prime score: credit + income + tenure + verified income
    prime = (
        (cluster_means["credit_score"] / 850) * 0.30
        + (cluster_means["income"] / 200000) * 0.30
        + (cluster_means["employment_years"] / 25) * 0.20
        + cluster_means["verified_income"] * 0.20
    )
    # Risk score: low credit + high DTI
    risk = (
        (1 - cluster_means["credit_score"] / 850) * 0.60
        + cluster_means["debt_to_income"] * 0.40
    )

    by_prime = prime.sort_values(ascending=False)
    cluster_to_seg = {}
    cluster_to_seg[by_prime.index[0]] = 2  # Established Prime
    cluster_to_seg[by_prime.index[1]] = 1  # Rising Prime

    remaining = list(by_prime.index[2:])
    by_risk = risk.loc[remaining].sort_values(ascending=False)
    cluster_to_seg[by_risk.index[0]] = 3  # Subprime High-Risk
    cluster_to_seg[by_risk.index[1]] = 0  # Mass Market

    return np.array([cluster_to_seg[c] for c in labels])


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns

    df = generate_customer_data()
    df = build_features(df)
    feats = get_feature_columns()
    result = run_clustering(df, feats)
    print("Silhouette:", result["silhouette_score"])
    for seg_id, prof in result["profiles"].items():
        print(f"  Cluster {seg_id}: {prof['name']} ({prof['count']} customers, {prof['pct']}%)")