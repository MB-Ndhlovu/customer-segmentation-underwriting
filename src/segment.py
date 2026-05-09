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
    """Elbow method + Silhouette analysis to select optimal k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k, inertias, silhouettes


def fit_kmeans(X_scaled, n_clusters=4):
    """Fit KMeans with k=4 (matching our 4 business segments)."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df, labels, feature_cols):
    """Build profiling summary per cluster."""
    df_temp = df.copy()
    df_temp["cluster"] = labels

    profiles = {}
    for cluster_id in sorted(df_temp["cluster"].unique()):
        subset = df_temp[df_temp["cluster"] == cluster_id][feature_cols]
        profiles[int(cluster_id)] = {
            "count": int((labels == cluster_id).sum()),
            "mean_income": float(subset["income"].mean()),
            "mean_credit_score": float(subset["credit_score"].mean()),
            "mean_employment_years": float(subset["employment_years"].mean()),
            "mean_dti": float(subset["debt_to_income"].mean()),
            "mean_loan_count": float(subset["loan_history_count"].mean()),
            "mean_age": float(subset["age"].mean()),
            "pct_homeowners": float(subset["home_ownership"].mean()),
            "pct_verified": float(subset["verified_income"].mean()),
        }

    return profiles
