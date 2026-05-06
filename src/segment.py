import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json


def find_optimal_k(X, k_range=range(2, 9)):
    """Elbow method + silhouette analysis."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return inertias, silhouettes


def segment_customers(X, n_clusters=4):
    """Fit KMeans with n_clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def profile_segments(df, labels, feature_cols):
    """Generate profile summary for each cluster."""
    df_temp = df.copy()
    df_temp['cluster'] = labels
    profiles = {}
    segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}

    for c in sorted(df_temp['cluster'].unique()):
        sub = df_temp[df_temp['cluster'] == c]
        profiles[int(c)] = {
            'name': segment_names.get(c, f'Segment {c}'),
            'size': int(len(sub)),
            'pct': round(len(sub) / len(df_temp) * 100, 1),
            'mean_income': round(float(sub['income'].mean()), 0),
            'mean_credit_score': round(float(sub['credit_score'].mean()), 1),
            'mean_employment_years': round(float(sub['employment_years'].mean()), 2),
            'mean_debt_to_income': round(float(sub['debt_to_income'].mean()), 3),
            'mean_loan_history_count': round(float(sub['loan_history_count'].mean()), 2),
            'mean_age': round(float(sub['age'].mean()), 1),
            'pct_homeowners': round(float(sub['home_ownership'].mean()) * 100, 1),
            'pct_verified_income': round(float(sub['verified_income'].mean()) * 100, 1),
        }
    return profiles


def run_segmentation(df, feature_cols):
    """Run full segmentation pipeline."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    # Evaluate silhouette across k range
    k_range = range(2, 9)
    inertias, silhouettes = find_optimal_k(X, k_range)
    best_k = list(k_range)[np.argmax(silhouettes)]
    best_sil = max(silhouettes)

    # Final clustering with k=4 (business requirement)
    labels, km = segment_customers(X, n_clusters=4)
    sil_score = silhouette_score(X, labels)

    # Profile
    profiles = profile_segments(df, labels, feature_cols)

    return {
        'labels': labels,
        'model': km,
        'scaler': scaler,
        'silhouette_score': round(sil_score, 4),
        'best_k_from_analysis': int(best_k),
        'best_silhouette_found': round(best_sil, 4),
        'elbow_inertias': [round(float(i), 1) for i in inertias],
        'silhouettes_by_k': [round(float(s), 4) for s in silhouettes],
        'profiles': profiles
    }