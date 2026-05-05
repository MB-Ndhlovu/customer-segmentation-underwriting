import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    return list(k_range), inertias, silhouettes

def cluster_customers(df, feature_cols, n_clusters=4):
    scaler = StandardScaler()
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X_scaled)
    
    return df, km, scaler

def map_clusters_to_segments(df, feature_cols):
    df = df.copy()
    
    # Compute cluster centroids in original space for profiling
    cluster_profiles = df.groupby('cluster')[feature_cols].mean()
    
    # Risk score: higher = worse risk
    # Components: DTI (bad), credit_norm (good), income_norm (good)
    cluster_profiles['credit_norm'] = (cluster_profiles['credit_score'] - 300) / 550
    cluster_profiles['income_norm'] = cluster_profiles['income'] / cluster_profiles['income'].max()
    
    # Risk = high DTI + low credit + low income
    cluster_profiles['risk_score'] = (
        cluster_profiles['debt_to_income'] 
        - 0.5 * cluster_profiles['credit_norm']
        - 0.3 * cluster_profiles['income_norm']
    )
    
    # Highest risk score -> highest label (3 = Subprime High-Risk)
    cluster_order = cluster_profiles.sort_values('risk_score', ascending=True).index.tolist()
    mapping = {c: i for i, c in enumerate(cluster_order)}
    
    df['segment_label'] = df['cluster'].map(mapping)
    df['segment_name'] = df['segment_label'].map(SEGMENT_NAMES)
    
    return df

def profile_segments(df, feature_cols):
    profiles = {}
    
    for seg_id in sorted(df['segment_label'].unique()):
        seg_df = df[df['segment_label'] == seg_id]
        seg_name = SEGMENT_NAMES[seg_id]
        
        profiles[seg_name] = {
            'count': int(len(seg_df)),
            'pct': round(len(seg_df) / len(df) * 100, 1),
            'features': {
                col: round(seg_df[col].mean(), 2) for col in feature_cols
            }
        }
    
    return profiles

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import engineer_features, FEATURE_COLS
    
    df = generate_customer_data(5000)
    df = engineer_features(df)
    df, km, scaler = cluster_customers(df, FEATURE_COLS)
    
    ks, inertias, silhouettes = find_optimal_k(df[FEATURE_COLS])
    print("K values:", ks)
    print("Silhouettes:", [round(s, 3) for s in silhouettes])
    
    df = map_clusters_to_segments(df, FEATURE_COLS)
    profiles = profile_segments(df, FEATURE_COLS)
    
    for name, prof in profiles.items():
        print(f"\n{name} ({prof['pct']}%):")
        print(f"  Count: {prof['count']}")
        print(f"  Avg Credit Score: {prof['features']['credit_score']}")
        print(f"  Avg Income: {prof['features']['income']:.0f}")