"""KMeans clustering with silhouette analysis, Elbow method, and segment profiling."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json

SEGMENT_LABELS = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

def find_best_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """Run Elbow + silhouette analysis to pick best k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    silhouette_scores = dict(zip(k_range, silhouettes))
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    return {"best_k": best_k, "silhouette_scores": silhouette_scores, "inertias": dict(zip(k_range, inertias))}


def assign_segment_labels(df: pd.DataFrame, feature_cols: list, n_clusters: int = 4) -> pd.DataFrame:
    """Fit KMeans, assign segment labels (0-3), and profile each segment."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette analysis
    analysis = find_best_k(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment_label"] = km.fit_predict(X_scaled)

    # Map kmeans cluster IDs to business labels by ascending median income:
    # lowest income cluster  → label 3 (Subprime High-Risk)
    # next                   → label 0 (Mass Market)
    # next                   → label 1 (Rising Prime)
    # highest income cluster → label 2 (Established Prime)
    segment_medians = (
        df.groupby("segment_label")["income"]
        .median()
        .sort_values()
        .reset_index()
        .rename(columns={"segment_label": "cluster_id"})
    )
    # Assign business labels in income order: 3, 0, 1, 2
    business_labels = [3, 0, 1, 2]
    label_map = {
        row["cluster_id"]: business_labels[i]
        for i, (_, row) in enumerate(segment_medians.iterrows())
    }
    df["segment_label"] = df["segment_label"].map(label_map)

    sil = silhouette_score(X_scaled, df["segment_label"])

    # Profiles
    profiles = {}
    for seg in sorted(df["segment_label"].unique()):
        sub = df[df["segment_label"] == seg]
        profiles[int(seg)] = {
            "name": SEGMENT_LABELS[seg],
            "count": int(len(sub)),
            "pct": round(len(sub) / len(df) * 100, 1),
            "income_mean": round(float(sub["income"].mean()), 2),
            "income_median": round(float(sub["income"].median()), 2),
            "credit_score_mean": round(float(sub["credit_score"].mean()), 1),
            "employment_years_mean": round(float(sub["employment_years"].mean()), 2),
            "debt_to_income_mean": round(float(sub["debt_to_income"].mean()), 3),
            "loan_history_count_mean": round(float(sub["loan_history_count"].mean()), 2),
            "age_mean": round(float(sub["age"].mean()), 1),
            "verified_income_pct": round(sub["verified_income"].mean() * 100, 1),
        }

    return df, {
        "best_k_from_analysis": analysis["best_k"],
        "silhouette_at_k4": round(sil, 4),
        "silhouette_scores": {str(k): round(v, 4) for k, v in analysis["silhouette_scores"].items()},
        "segment_profiles": profiles,
    }