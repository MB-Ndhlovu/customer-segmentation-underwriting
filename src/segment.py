import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> dict:
    """Elbow method + silhouette analysis to find optimal k."""
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
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k": int(optimal_k),
        "best_silhouette": float(max(silhouettes)),
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> tuple:
    """Fit KMeans and return labels + model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Profile each segment with key statistics."""
    df_labeled = df.copy()
    df_labeled["segment"] = labels

    profiles = {}
    for seg in sorted(df_labeled["segment"].unique()):
        seg_data = df_labeled[df_labeled["segment"] == seg]
        profiles[int(seg)] = {
            "name": SEGMENT_NAMES.get(seg, f"Segment {seg}"),
            "size": int(len(seg_data)),
            "pct": round(len(seg_data) / len(df_labeled) * 100, 1),
            "mean_income": round(float(seg_data["income"].mean()), 0),
            "mean_credit_score": round(float(seg_data["credit_score"].mean()), 1),
            "mean_employment_years": round(float(seg_data["employment_years"].mean()), 2),
            "mean_debt_to_income": round(float(seg_data["debt_to_income"].mean()), 3),
            "mean_loan_history_count": round(float(seg_data["loan_history_count"].mean()), 2),
            "mean_age": round(float(seg_data["age"].mean()), 1),
            "home_ownership_rate": round(float(seg_data["home_ownership"].mean()), 3),
            "verified_income_rate": round(float(seg_data["verified_income"].mean()), 3),
        }

    return profiles


def assign_segment_names(labels: np.ndarray) -> np.ndarray:
    """Map cluster IDs to meaningful segment names via label encoding by profile."""
    return labels


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns

    df = generate_customer_data()
    X = build_features(df)
    feature_cols = get_feature_columns()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    k_results = find_optimal_k(X_scaled)
    print("Optimal K:", k_results["optimal_k"])
    print("Silhouettes:", k_results["silhouettes"])

    labels, km = fit_kmeans(X_scaled, n_clusters=4)
    profiles = profile_segments(X, labels)
    for seg, profile in profiles.items():
        print(f"\nSegment {seg} ({profile['name']}):")
        print(f"  Size: {profile['size']} ({profile['pct']}%)")
        print(f"  Income: {profile['mean_income']}, Credit: {profile['mean_credit_score']}")
