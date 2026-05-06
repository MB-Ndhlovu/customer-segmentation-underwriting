"""KMeans clustering with Elbow method, Silhouette analysis, and segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import json


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Run Elbow + Silhouette analysis; return best k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    best_k = list(k_range)[np.argmax(silhouettes)]
    return best_k, inertias, silhouettes


def cluster(X_scaled, n_clusters=4):
    """Fit KMeans and return labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def profile_segments(df, labels, feature_cols):
    """Produce summary statistics for each segment."""
    df_temp = df.copy()
    df_temp['segment'] = labels
    profiles = {}
    segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}

    for seg_id in sorted(df_temp['segment'].unique()):
        sub = df_temp[df_temp['segment'] == seg_id]
        profiles[int(seg_id)] = {
            'name': segment_names.get(seg_id, f'Segment {seg_id}'),
            'count': int(len(sub)),
            'pct': round(len(sub) / len(df_temp) * 100, 1),
            'mean_income': round(float(sub['income'].mean()), 2),
            'mean_credit_score': round(float(sub['credit_score'].mean()), 1),
            'mean_debt_to_income': round(float(sub['debt_to_income'].mean()), 4),
            'mean_employment_years': round(float(sub['employment_years'].mean()), 2),
            'mean_loan_history_count': round(float(sub['loan_history_count'].mean()), 2),
            'pct_homeowners': round(float(sub['home_ownership'].mean()) * 100, 1),
            'pct_verified_income': round(float(sub['verified_income'].mean()) * 100, 1),
        }

    # Map cluster IDs to meaningful names based on characteristics
    # Sort by income desc to assign labels: highest income = Established Prime, etc.
    seg_by_income = sorted(profiles.keys(), key=lambda k: profiles[k]['mean_income'], reverse=True)
    label_map = {
        seg_by_income[0]: 'Established Prime',   # highest income
        seg_by_income[1]: 'Rising Prime',          # second highest income
        seg_by_income[2]: 'Mass Market',           # middle
        seg_by_income[3]: 'Subprime High-Risk',    # lowest income / highest DTI
    }

    return profiles, label_map


def save_results(profiles, seg_summary, silhouette_avg, report_path):
    """Save segmentation results to JSON."""
    results = {
        'segments': {str(k): {**v, 'label': seg_summary.get(k, 'Unknown')} for k, v in profiles.items()},
        'silhouette_score': round(float(silhouette_avg), 4),
        'n_segments': len(profiles),
    }
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results