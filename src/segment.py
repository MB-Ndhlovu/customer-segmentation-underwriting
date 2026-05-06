import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

def find_optimal_k(X_scaled, k_range=range(2, 10)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return dict(zip(k_range, inertias)), dict(zip(k_range, silhouettes))

def cluster_customers(X_scaled, n_clusters=4, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km

def profile_segments(X_df, labels, feature_cols):
    df_temp = X_df.copy()
    df_temp['cluster'] = labels
    profiles = df_temp.groupby('cluster')[feature_cols].mean().to_dict('index')

    segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}
    results = {}
    for cid in sorted(profiles.keys()):
        p = profiles[cid]
        results[segment_names.get(cid, f'Cluster {cid}')] = {
            'cluster_id': int(cid),
            'mean_income': float(p.get('income', 0)),
            'mean_credit_score': float(p.get('credit_score', 0)),
            'mean_employment_years': float(p.get('employment_years', 0)),
            'mean_debt_to_income': float(p.get('debt_to_income', 0)),
            'mean_loan_history_count': float(p.get('loan_history_count', 0)),
            'mean_age': float(p.get('age', 0)),
            'home_ownership_rate': float(p.get('home_ownership', 0)),
            'verified_income_rate': float(p.get('verified_income', 0)),
        }
    return results

def run_segmentation(X_scaled, X_df, feature_cols, n_clusters=4):
    labels, km = cluster_customers(X_scaled, n_clusters=n_clusters)
    sil = silhouette_score(X_scaled, labels)
    profiles = profile_segments(X_df, labels, feature_cols)

    return {
        'labels': labels,
        'model': km,
        'silhouette_score': float(sil),
        'profiles': profiles,
    }