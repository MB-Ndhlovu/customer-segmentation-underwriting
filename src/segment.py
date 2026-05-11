import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return inertias, silhouettes


def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df, labels, feature_names):
    profiles = {}
    for seg in np.unique(labels):
        mask = labels == seg
        seg_data = df[mask]
        profiles[int(seg)] = {
            'count': int(seg_data.shape[0]),
            'pct': round(seg_data.shape[0] / len(labels) * 100, 1),
            'mean_income': round(seg_data['income'].mean(), 0),
            'mean_credit_score': round(seg_data['credit_score'].mean(), 1),
            'mean_employment_years': round(seg_data['employment_years'].mean(), 2),
            'mean_debt_to_income': round(seg_data['debt_to_income'].mean(), 3),
            'mean_loan_history_count': round(seg_data['loan_history_count'].mean(), 2),
            'mean_age': round(seg_data['age'].mean(), 1),
            'home_ownership_rate': round(seg_data['home_ownership'].mean(), 3),
            'verified_income_rate': round(seg_data['verified_income'].mean(), 3),
        }
    return profiles


def assign_segment_names(profiles):
    """
    Heuristic naming based on credit_score and income order.
    Returns dict mapping 0-3 to names.
    """
    segs = list(profiles.keys())
    # Sort by credit_score descending
    sorted_segs = sorted(segs, key=lambda s: profiles[s]['mean_credit_score'], reverse=True)

    names = {
        sorted_segs[0]: 'Established Prime',      # highest credit
        sorted_segs[1]: 'Rising Prime',           # 2nd highest
        sorted_segs[2]: 'Mass Market',            # middle
        sorted_segs[3]: 'Subprime High-Risk',     # lowest credit
    }
    return names