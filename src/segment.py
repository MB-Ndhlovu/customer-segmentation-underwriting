import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

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

def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels

def profile_segments(X, labels, df_orig):
    profiles = {}
    for seg in np.unique(labels):
        mask = labels == seg
        seg_data = X[mask]
        orig_seg_data = df_orig[mask]
        profiles[int(seg)] = {
            "n": int(mask.sum()),
            "mean_income": float(seg_data["income"].mean()),
            "mean_credit_score": float(seg_data["credit_score"].mean()),
            "mean_employment_years": float(seg_data["employment_years"].mean()),
            "mean_dti": float(seg_data["debt_to_income"].mean()),
            "mean_loan_count": float(seg_data["loan_history_count"].mean()),
            "mean_age": float(seg_data["age"].mean()),
            "pct_homeowners": float(orig_seg_data["home_ownership"].mean()),
            "pct_verified_income": float(orig_seg_data["verified_income"].mean()),
        }
    return profiles

def assign_segment_names(labels):
    return np.array([SEGMENT_NAMES[l] for l in labels])

if __name__ == "__main__":
    from data_loader import load_data
    from features import build_features, get_feature_names
    from sklearn.preprocessing import StandardScaler

    df = load_data()
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km, labels = fit_kmeans(X_scaled, 4)
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil:.4f}")
    profiles = profile_segments(X, labels, df)
    for seg, p in profiles.items():
        print(f"Segment {seg} ({SEGMENT_NAMES[seg]}): n={p['n']}, "
              f"income={p['mean_income']:.0f}, credit={p['mean_credit_score']:.0f}, "
              f"DTI={p['mean_dti']:.3f}, verified={p['pct_verified_income']:.2%}")
