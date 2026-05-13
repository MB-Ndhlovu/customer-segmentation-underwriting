import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_k(X_scaled, k_range=range(2, 10)):
    """Elbow method and silhouette analysis"""
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

    optimal_k = k_range[np.argmax(silhouettes)]
    return optimal_k, inertias, silhouettes

def segment_customers(df, n_clusters=4):
    """Perform KMeans clustering and assign segment labels"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, scaler, kmeans

def profile_segments(df, labels):
    """Profile each segment with statistics"""
    df_with_labels = df.copy()
    df_with_labels['segment'] = labels

    segment_names = {
        0: 'Mass Market',
        1: 'Rising Prime',
        2: 'Established Prime',
        3: 'Subprime High-Risk'
    }

    profiles = {}
    for seg in range(4):
        seg_data = df_with_labels[df_with_labels['segment'] == seg]
        profiles[segment_names[seg]] = {
            'count': len(seg_data),
            'pct': len(seg_data) / len(df_with_labels) * 100,
            'mean_income': seg_data['income'].mean(),
            'mean_credit_score': seg_data['credit_score'].mean(),
            'mean_employment_years': seg_data['employment_years'].mean(),
            'mean_debt_to_income': seg_data['debt_to_income'].mean(),
            'mean_loan_history_count': seg_data['loan_history_count'].mean(),
            'mean_age': seg_data['age'].mean(),
            'pct_home_owners': seg_data['home_ownership'].mean() * 100,
            'pct_verified_income': seg_data['verified_income'].mean() * 100,
        }

    return profiles, df_with_labels

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_feature_matrix

    df = generate_synthetic_data()
    features = build_feature_matrix(df)

    labels, scaler, kmeans = segment_customers(features)
    profiles, df_labelled = profile_segments(df, labels)

    for name, stats in profiles.items():
        print(f"\n{name}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")