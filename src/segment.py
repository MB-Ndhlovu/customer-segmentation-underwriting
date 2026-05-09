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

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes

def cluster(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km

def profile_segments(X, labels):
    profile = X.copy()
    profile["segment_label"] = labels
    summary = profile.groupby("segment_label").mean()
    named = {}
    for idx in summary.index:
        seg_id = int(idx)
        named[seg_id] = {
            "name": SEGMENT_NAMES.get(seg_id, f"Segment {seg_id}"),
            "mean_income": float(summary.loc[idx, "income"]),
            "mean_credit_score": float(summary.loc[idx, "credit_score"]),
            "mean_employment_years": float(summary.loc[idx, "employment_years"]),
            "mean_debt_to_income": float(summary.loc[idx, "debt_to_income"]),
            "mean_loan_history_count": float(summary.loc[idx, "loan_history_count"]),
            "mean_age": float(summary.loc[idx, "age"]),
            "pct_homeowner": float(summary.loc[idx, "home_ownership"]),
            "pct_verified_income": float(summary.loc[idx, "verified_income"]),
            "count": int((labels == seg_id).sum()),
        }
    return named