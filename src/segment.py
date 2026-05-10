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


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Elbow method + silhouette analysis to find optimal k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    optimal_k = list(k_range)[np.argmax(silhouettes)]
    return optimal_k, inertias, silhouettes


def fit_kmeans(X_scaled, n_clusters=4):
    """Fit KMeans with k=4."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df, labels):
    """Profile each cluster with mean feature values."""
    df_plot = df.copy()
    df_plot["cluster"] = labels

    profiles = {}
    for cluster_id in sorted(df_plot["cluster"].unique()):
        subset = df_plot[df_plot["cluster"] == cluster_id]
        profiles[int(cluster_id)] = {
            "count": int(len(subset)),
            "pct": round(len(subset) / len(df_plot) * 100, 2),
            "mean_income": round(subset["income"].mean(), 2),
            "mean_credit_score": round(subset["credit_score"].mean(), 1),
            "mean_employment_years": round(subset["employment_years"].mean(), 2),
            "mean_debt_to_income": round(subset["debt_to_income"].mean(), 4),
            "mean_loan_history_count": round(subset["loan_history_count"].mean(), 2),
            "mean_age": round(subset["age"].mean(), 1),
            "home_ownership_rate": round(subset["home_ownership_status"].mean(), 3),
            "verified_income_rate": round(subset["verified_income"].mean(), 3),
        }

    # Label segments based on profiles
    label_map = assign_segment_labels(profiles)
    return profiles, label_map


def assign_segment_labels(profiles):
    """Map cluster IDs to human-readable segment names."""
    # Identify highest credit score cluster as Established Prime
    # Identify lowest credit score + highest DTI as Subprime High-Risk
    # Identify mid-level as Rising Prime or Mass Market
    credit_scores = {k: v["mean_credit_score"] for k, v in profiles.items()}
    dti_scores = {k: v["mean_debt_to_income"] for k, v in profiles.items()}

    sorted_by_credit = sorted(credit_scores, key=credit_scores.get, reverse=True)
    sorted_by_dti = sorted(dti_scores, key=dti_scores.get, reverse=True)

    label_map = {}
    # Highest credit = Established Prime
    label_map[sorted_by_credit[0]] = "Established Prime"
    # Highest DTI + lower credit = Subprime High-Risk
    label_map[sorted_by_dti[0]] = "Subprime High-Risk"

    remaining = [k for k in profiles if k not in label_map]
    if len(remaining) == 2:
        # Higher income = Rising Prime, lower = Mass Market
        incomes = {k: profiles[k]["mean_income"] for k in remaining}
        sorted_by_income = sorted(remaining, key=incomes.get, reverse=True)
        label_map[sorted_by_income[0]] = "Rising Prime"
        label_map[sorted_by_income[1]] = "Mass Market"

    return label_map


def run_segmentation(df, feature_cols):
    """Run full segmentation pipeline."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_k, inertias, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 8))

    # Force k=4 for business alignment
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil_score = silhouette_score(X_scaled, labels)

    profiles, label_map = profile_segments(df, labels)

    # Map labels to segment names
    segment_names = [label_map[l] for l in labels]

    results = {
        "optimal_k_found": int(optimal_k),
        "k_used": 4,
        "silhouette_score": round(sil_score, 4),
        "inertias": [round(i, 2) for i in inertias],
        "silhouettes": [round(s, 4) for s in silhouettes],
        "profiles": profiles,
        "label_map": {str(k): v for k, v in label_map.items()},
        "segment_names": segment_names,
    }

    return labels, scaler, results


def save_results(results, path="reports/segmentation_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


import os