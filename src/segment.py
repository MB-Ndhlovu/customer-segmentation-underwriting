import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import json
import os


def find_optimal_k(X, k_range=range(2, 11)):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    return list(k_range), inertias, silhouettes


def elbow_analysis(k_range, inertias):
    """Simple elbow detection — second derivative."""
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = int(np.argmax(second_diffs) + 2)
    return elbow_idx


def run_kmeans(X, n_clusters=4, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def silhouette_analysis(X, labels):
    score = silhouette_score(X, labels)
    sample_scores = silhouette_samples(X, labels)
    return score, sample_scores


def profile_segments(df, labels, feature_cols):
    """Build segment profiles."""
    df_labeled = df.copy()
    df_labeled['segment'] = labels

    profiles = {}
    for seg in sorted(df_labeled['segment'].unique()):
        seg_df = df_labeled[df_labeled['segment'] == seg]
        profiles[int(seg)] = {
            'count': int(len(seg_df)),
            'pct': round(len(seg_df) / len(df_labeled) * 100, 1),
            'mean_income': round(float(seg_df['income'].mean()), 2),
            'mean_credit_score': round(float(seg_df['credit_score'].mean()), 1),
            'mean_dti': round(float(seg_df['debt_to_income'].mean()), 3),
            'mean_loan_count': round(float(seg_df['loan_history_count'].mean()), 2),
            'mean_age': round(float(seg_df['age'].mean()), 1),
            'pct_homeowners': round(float((seg_df['home_ownership_status'] >= 1).mean()) * 100, 1),
            'pct_verified_income': round(float(seg_df['verified_income'].mean() * 100), 1),
        }

    # Assign business labels based on characteristics
    seg_assignments = {}
    for seg_id, prof in profiles.items():
        if prof['mean_credit_score'] < 600:
            seg_assignments[seg_id] = 'Subprime High-Risk'
        elif prof['mean_income'] > 90000:
            seg_assignments[seg_id] = 'Established Prime'
        elif prof['mean_income'] > 50000:
            seg_assignments[seg_id] = 'Rising Prime'
        else:
            seg_assignments[seg_id] = 'Mass Market'

    return profiles, seg_assignments


def save_results(profiles, seg_assignments, silhouette_scores, n_clusters, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_scores,
        'segment_profiles': profiles,
        'segment_labels': seg_assignments,
        'pipeline_summary': {
            'total_customers': sum(p['count'] for p in profiles.values()),
            'segments': list(profiles.keys())
        }
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results