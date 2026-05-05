"""KMeans clustering for customer segmentation with elbow method and silhouette analysis."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json


SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}


def find_optimal_k(X, k_range=range(2, 10)):
    """Use elbow method and silhouette analysis to find optimal cluster count."""
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k, inertias, silhouettes


def fit_kmeans(X, n_clusters=4):
    """Fit KMeans with specified cluster count."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def profile_segments(df, labels, feature_cols):
    """Profile each segment with mean feature values."""
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    profiles = {}
    for cluster_id in sorted(df_copy['cluster'].unique()):
        segment_df = df_copy[df_copy['cluster'] == cluster_id]
        profiles[int(cluster_id)] = {
            'size': int(len(segment_df)),
            'pct': round(len(segment_df) / len(df_copy) * 100, 1),
            **{feat: round(float(segment_df[feat].mean()), 4) for feat in feature_cols}
        }
    
    return profiles


def map_clusters_to_segments(profiles):
    """Map cluster IDs to business segment names based on feature profiles."""
    # Score each cluster on risk/profile dimensions
    cluster_scores = {}
    for cid, profile in profiles.items():
        # High credit + high income + low DTI + homeowner = Prime
        score = (
            profile['credit_score'] / 850 * 0.3 +
            profile['income'] / 180000 * 0.3 +
            (1 - profile['debt_to_income']) * 0.25 +
            profile['home_ownership'] * 0.15
        )
        cluster_scores[cid] = score
    
    # Sort by score to assign labels
    sorted_clusters = sorted(cluster_scores, key=cluster_scores.get)
    
    # Assign: 0=Mass Market, 1=Rising Prime, 2=Established Prime, 3=Subprime High-Risk
    # But we want Established Prime to be highest score, Mass Market mid, Subprime lowest
    # So reverse: highest score = 2 (Established Prime), lowest = 3 (Subprime High-Risk)
    assignment = {}
    assignment[sorted_clusters[-1]] = 2  # Established Prime (highest)
    assignment[sorted_clusters[-2]] = 1  # Rising Prime
    assignment[sorted_clusters[1]] = 0   # Mass Market
    assignment[sorted_clusters[0]] = 3    # Subprime High-Risk (lowest)
    
    return assignment


def save_results(profiles, silhouette, optimal_k, assignment, output_path):
    """Save segmentation results to JSON."""
    results = {
        'optimal_k': int(optimal_k),
        'silhouette_score': round(float(silhouette), 4),
        'cluster_assignment': {str(k): v for k, v in assignment.items()},
        'segment_profiles': {str(k): v for k, v in profiles.items()},
        'segment_names': SEGMENT_NAMES,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features, scale_features
    
    df = generate_customer_data()
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income'
    ]
    X_raw = df[feature_cols]
    X_scaled, scaler = scale_features(X_raw)
    
    optimal_k, inertias, silhouettes = find_optimal_k(X_scaled)
    print(f"Optimal k: {optimal_k}, Silhouette: {silhouettes[optimal_k-2]:.4f}")
    
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    profiles = profile_segments(df, labels, feature_cols)
    assignment = map_clusters_to_segments(profiles)
    
    print("\nSegment assignment:", assignment)
    for cid, profile in profiles.items():
        print(f"Cluster {cid}: n={profile['size']}, credit_score={profile['credit_score']:.0f}, income={profile['income']:.0f}")