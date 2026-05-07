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

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    elbow = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        elbow.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    best_k = list(k_range)[np.argmax(silhouettes)]
    return elbow, silhouettes, best_k

def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return km, labels, score

def remap_labels_to_segments(df, labels, feature_cols):
    df = df.copy()
    df["_raw_cluster"] = labels
    means = df.groupby("_raw_cluster")[feature_cols].mean()
    credit_means = means["credit_score"].sort_values()
    credit_order = credit_means.index.tolist()

    target_order = [3, 0, 1, 2]
    mapping = {raw: target for raw, target in zip(credit_order, target_order)}
    new_labels = np.array([mapping[l] for l in labels])
    return new_labels, mapping

def map_segments_to_names(labels):
    return [SEGMENT_LABELS[l] for l in labels]

def profile_segments(df, labels):
    df = df.copy()
    df["segment_label"] = labels
    profiles = {}
    for seg_idx in sorted(df["segment_label"].unique()):
        seg_data = df[df["segment_label"] == seg_idx]
        name = SEGMENT_LABELS[seg_idx]
        profiles[name] = {
            "count": int(len(seg_data)),
            "pct": round(len(seg_data) / len(df) * 100, 1),
            "income_mean": round(float(seg_data["income"].mean()), 2),
            "credit_score_mean": round(float(seg_data["credit_score"].mean()), 1),
            "debt_to_income_mean": round(float(seg_data["debt_to_income"].mean()), 3),
            "employment_years_mean": round(float(seg_data["employment_years"].mean()), 2),
            "age_mean": round(float(seg_data["age"].mean()), 1),
            "home_ownership_rate": round(float(seg_data["home_ownership"].mean()), 3),
            "verified_income_rate": round(float(seg_data["verified_income"].mean()), 3),
        }
    return profiles

def save_results(profiles, silhouette, elbow, silhouettes_by_k, best_k, path):
    out = {
        "best_k": best_k,
        "silhouette_score": round(silhouette, 4),
        "elbow_inertias": [round(e, 2) for e in elbow],
        "silhouettes_by_k": silhouettes_by_k if isinstance(silhouettes_by_k, dict) else {str(k): round(s, 4) for k, s in silhouettes_by_k},
        "segment_profiles": profiles,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")