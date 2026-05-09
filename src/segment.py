"""
Segmentation using KMeans clustering.
- Elbow method to estimate optimal k
- Silhouette analysis to validate cluster quality
- Segment profiling for interpretability
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import json


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)):
    """
    Run elbow method and silhouette analysis.
    Returns dict with inertias and silhouette scores per k.
    """
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(float(km.inertia_))
        results["silhouette"].append(float(silhouette_score(X_scaled, labels)))

    return results


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """Fit KMeans with specified cluster count."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray, segment_names: dict):
    """
    Compute profile statistics for each segment.
    Returns list of segment profile dicts.
    """
    df_labeled = df.copy()
    df_labeled["segment"] = labels

    profiles = []
    for seg_id in sorted(df_labeled["segment"].unique()):
        subset = df_labeled[df_labeled["segment"] == seg_id]
        profile = {
            "segment_id": int(seg_id),
            "segment_name": segment_names.get(seg_id, f"Segment {seg_id}"),
            "size": int(len(subset)),
            "pct": round(len(subset) / len(df_labeled) * 100, 1),
            "mean_income": round(float(subset["income"].mean()), 2),
            "mean_credit_score": round(float(subset["credit_score"].mean()), 1),
            "mean_employment_years": round(float(subset["employment_years"].mean()), 2),
            "mean_debt_to_income": round(float(subset["debt_to_income"].mean()), 4),
            "mean_loan_history_count": round(float(subset["loan_history_count"].mean()), 2),
            "mean_age": round(float(subset["age"].mean()), 1),
            "home_ownership_rate": round(float(subset["home_ownership"].mean()), 3),
            "verified_income_rate": round(float(subset["verified_income"].mean()), 3),
        }
        profiles.append(profile)

    return profiles


def save_results(profiles, optimal_k, silhouette_scores, path: str):
    """Save segmentation results to JSON."""
    results = {
        "optimal_k": optimal_k,
        "silhouette_scores_by_k": silhouette_scores,
        "segment_profiles": profiles
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")