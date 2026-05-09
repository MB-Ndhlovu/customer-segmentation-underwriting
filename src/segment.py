"""KMeans clustering, elbow method, silhouette analysis, and segment profiling."""

import json
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
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbls))
    return inertias, silhouettes


def profile_segment(df, labels, seg_id):
    mask = labels == seg_id
    seg_df = df.loc[mask]
    return {
        "segment_id": seg_id,
        "segment_name": SEGMENT_NAMES[seg_id],
        "count": int(mask.sum()),
        "pct": round(mask.sum() / len(labels) * 100, 1),
        "income_mean": round(float(seg_df["income"].mean()), 2),
        "credit_score_mean": round(float(seg_df["credit_score"].mean()), 1),
        "employment_years_mean": round(float(seg_df["employment_years"].mean()), 2),
        "debt_to_income_mean": round(float(seg_df["debt_to_income"].mean()), 4),
        "age_mean": round(float(seg_df["age"].mean()), 1),
        "home_ownership_mode": int(seg_df["home_ownership"].mode().iloc[0]),
        "verified_income_pct": round(float(seg_df["verified_income"].mean()) * 100, 1),
    }


def run_segmentation(df, feature_cols, n_clusters=4):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, silhouettes = find_optimal_k(X_scaled)

    best_k_idx = int(np.argmax(silhouettes[-(9 - n_clusters):])) + (n_clusters - 2)
    best_silhouette = round(silhouettes[best_k_idx], 4)
    best_k = list(range(2, 9))[best_k_idx]

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = round(silhouette_score(X_scaled, labels), 4)

    profiles = [profile_segment(df, labels, i) for i in range(n_clusters)]

    return {
        "k_used": n_clusters,
        "inertias": [round(float(i), 2) for i in inertias],
        "silhouettes": [round(float(s), 4) for s in silhouettes],
        "best_k_by_silhouette": int(best_k),
        "silhouette_at_best_k": best_silhouette,
        "silhouette_at_k4": sil,
        "cluster_labels": labels.tolist(),
        "profiles": profiles,
    }