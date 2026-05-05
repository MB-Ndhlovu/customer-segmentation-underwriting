import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk"
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> dict:
    """
    Elbow method + Silhouette analysis to select optimal k.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    optimal_k = k_range[np.argmax(silhouettes)]
    return {
        "k_range": list(k_range),
        "inertias": [float(i) for i in inertias],
        "silhouettes": [float(s) for s in silhouettes],
        "optimal_k": int(optimal_k),
        "best_silhouette": float(max(silhouettes))
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def silhouette_analysis(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    return silhouette_score(X_scaled, labels)


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Profile each segment with mean feature values.
    """
    profile_df = df.copy()
    profile_df["segment_label"] = labels

    segment_profiles = profile_df.groupby("segment_label").mean(numeric_only=True)
    segment_profiles["count"] = profile_df.groupby("segment_label").size()

    # Map segment names
    segment_profiles["segment_name"] = segment_profiles.index.map(
        lambda x: SEGMENT_NAMES.get(x, f"Segment {x}")
    )

    return segment_profiles


def map_labels_to_business_segments(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Map KMeans cluster IDs to business-meaningful segment names based on characteristic analysis.
    We identify each cluster by its centroid proximity to known segment profiles.
    """
    profile_df = df.copy()
    profile_df["segment_label"] = labels

    seg_means = profile_df.groupby("segment_label").mean(numeric_only=True)

    # Score each cluster against known segment characteristics
    scores = {}
    for cluster_id in seg_means.index:
        row = seg_means.loc[cluster_id]
        scores[cluster_id] = {
            "credit_score": row["credit_score"],
            "income": row["income"],
            "debt_to_income": row["debt_to_income"],
            "employment_years": row["employment_years"],
            "age": row["age"],
        }

    # Rank clusters by credit_score + income descending -> assign prime first
    ranked = sorted(scores.keys(), key=lambda c: (
        scores[c]["credit_score"] + scores[c]["income"] / 1000
    ), reverse=True)

    mapping = {
        ranked[0]: 2,  # Established Prime (highest credit + income)
        ranked[1]: 1,  # Rising Prime
        ranked[2]: 0,  # Mass Market
        ranked[3]: 3,  # Subprime High-Risk (lowest credit + income)
    }

    return np.array([mapping[l] for l in labels])


def run_segmentation(X_scaled: np.ndarray, df: pd.DataFrame) -> dict:
    # Find optimal k
    opt = find_optimal_k(X_scaled)
    print(f"\nOptimal k by silhouette: {opt['optimal_k']} (score: {opt['best_silhouette']:.4f})")

    # Fit KMeans with k=4 (business requirement)
    km, raw_labels = fit_kmeans(X_scaled, n_clusters=4)

    # Map to business segments
    labels = map_labels_to_business_segments(raw_labels, df)

    sil = silhouette_analysis(X_scaled, labels)
    profiles = profile_segments(df, labels)

    print(f"\nSilhouette Score (k=4): {sil:.4f}")
    print("\nSegment Profiles:")
    print(profiles[["segment_name", "count", "income", "credit_score",
                     "debt_to_income", "employment_years", "age"]])

    result = {
        "optimal_k": opt["optimal_k"],
        "silhouette_raw": float(sil),
        "segment_counts": {
            SEGMENT_NAMES[i]: int((labels == i).sum())
            for i in range(4)
        },
        "segment_profiles": {
            SEGMENT_NAMES[i]: {
                col: float(profiles.loc[i, col]) if col in profiles.columns and i in profiles.index else None
                for col in ["income", "credit_score", "debt_to_income",
                           "employment_years", "age", "count"]
            }
            for i in range(4)
        },
        "cluster_inertia": float(km.inertia_)
    }

    return result


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, get_feature_matrix, scale_features
    from features import get_engineered_features

    df = generate_synthetic_data()
    X_raw = get_feature_matrix(df)
    X_scaled, scaler = scale_features(X_raw)
    result = run_segmentation(X_scaled, df)
    print(result)