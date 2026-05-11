import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership", "verified_income",
]

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """Elbow method + silhouette analysis."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    optimal_k = list(k_range)[np.argmax(silhouettes)]
    return optimal_k, inertias, silhouettes

def fit_kmeans(df, n_clusters=4):
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    inertia = km.inertia_
    df = df.copy()
    df["segment_label"] = labels
    return df, km, scaler, X_scaled, sil, inertia

def profile_segments(df):
    profiles = {}
    for seg in sorted(df["segment_label"].unique()):
        subset = df[df["segment_label"] == seg]
        profiles[int(seg)] = {
            "count": int(len(subset)),
            "pct": round(len(subset) / len(df) * 100, 1),
            "income_mean": round(subset["income"].mean(), 0),
            "credit_score_mean": round(subset["credit_score"].mean(), 1),
            "employment_years_mean": round(subset["employment_years"].mean(), 2),
            "debt_to_income_mean": round(subset["debt_to_income"].mean(), 4),
            "loan_history_count_mean": round(subset["loan_history_count"].mean(), 2),
            "age_mean": round(subset["age"].mean(), 1),
            "home_ownership_pct": round(subset["home_ownership"].mean() * 100, 1),
            "verified_income_pct": round(subset["verified_income"].mean() * 100, 1),
        }
    return profiles

def assign_segment_names(profiles):
    """Name segments based on financial profile medians."""
    # Sort by income_mean desc, credit_score desc
    sorted_segs = sorted(profiles.keys(), key=lambda s: (
        profiles[s]["income_mean"], profiles[s]["credit_score_mean"]
    ), reverse=True)

    names = {
        sorted_segs[0]: "Established Prime",
        sorted_segs[1]: "Rising Prime",
        sorted_segs[2]: "Mass Market",
        sorted_segs[3]: "Subprime High-Risk",
    }
    return names