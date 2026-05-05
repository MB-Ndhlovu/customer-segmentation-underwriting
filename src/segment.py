"""Clustering pipeline: KMeans, Elbow method, Silhouette analysis, Segment profiling."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership_enc", "verified_income_enc",
]


def run_elbow_silhouette(X_scaled: np.ndarray, k_range: range,) -> dict:
    """Compute inertia and silhouette scores for k in k_range."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X_scaled, labels)),
        }
    return results


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profile each cluster with mean feature values."""
    profile = df.groupby("segment_label")[FEATURE_COLS].mean().round(2)
    profile["count"] = df.groupby("segment_label").size()
    profile["segment_name"] = profile.index.map(lambda i: SEGMENT_NAMES.get(i, f"Cluster_{i}"))
    return profile


def assign_segment_names(labels: np.ndarray, profile_df: pd.DataFrame) -> np.ndarray:
    """Map cluster IDs to segment names based on credit_score ordering."""
    sorted_idx = profile_df.sort_values("credit_score").index.tolist()
    name_map = {
        sorted_idx[0]: "Subprime High-Risk",
        sorted_idx[1]: "Mass Market",
        sorted_idx[2]: "Rising Prime",
        sorted_idx[3]: "Established Prime",
    }
    return np.array([name_map.get(l, f"Cluster_{l}") for l in labels])


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Run full segmentation pipeline and return results."""
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_results = run_elbow_silhouette(X_scaled, range(2, 9))
    best_k = max(elbow_results.keys(), key=lambda k: elbow_results[k]["silhouette"])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    df_labels = df.copy()
    df_labels["cluster_label"] = labels

    profile = profile_clusters(df_labels, labels)
    named_labels = assign_segment_names(labels, profile)
    df_labels["segment_name"] = named_labels

    profile_dict = profile.reset_index().to_dict("records")
    for row in profile_dict:
        row["segment_name"] = SEGMENT_NAMES.get(row["segment_label"], f"Cluster_{row['segment_label']}")

    results = {
        "n_clusters": n_clusters,
        "best_k_by_silhouette": int(best_k),
        "silhouette_score": round(sil_score, 4),
        "elbow_results": {str(k): v for k, v in elbow_results.items()},
        "cluster_profiles": profile_dict,
        "cluster_counts": {int(k): int(v) for k, v in df_labels["cluster_label"].value_counts().to_dict().items()},
        "segment_mapping": {str(k): SEGMENT_NAMES[k] for k in range(n_clusters)},
    }

    return {
        "results": results,
        "labels": labels,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "km": km,
        "df_labeled": df_labels,
    }


if __name__ == "__main__":
    from data_loader import generate_customers
    from features import build_features

    df = generate_customers(5000)
    df = build_features(df)
    out = run_segmentation(df)
    print(json.dumps(out["results"], indent=2))