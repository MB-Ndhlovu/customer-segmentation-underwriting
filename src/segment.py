import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes


def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df, labels):
    profiles = {}
    for seg in np.unique(labels):
        mask = labels == seg
        seg_df = df[mask]
        profiles[int(seg)] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 1),
            "income_mean": float(seg_df["income"].mean()),
            "credit_score_mean": float(seg_df["credit_score"].mean()),
            "employment_years_mean": float(seg_df["employment_years"].mean()),
            "debt_to_income_mean": float(seg_df["debt_to_income"].mean()),
            "loan_history_count_mean": float(seg_df["loan_history_count"].mean()),
            "age_mean": float(seg_df["age"].mean()),
            "homeowner_pct": round(float(seg_df["home_ownership"].eq("own").mean() * 100), 1),
            "verified_income_pct": round(float(seg_df["verified_income"].mean() * 100), 1),
        }
    return profiles


def assign_segment_names(labels):
    return np.array([SEGMENT_NAMES[l] for l in labels])