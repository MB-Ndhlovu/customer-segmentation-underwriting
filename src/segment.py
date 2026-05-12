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
    3: "Subprime High-Risk"
}

def elbow_analysis(X_scaled, max_k=10):
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias

def find_optimal_k(X_scaled, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k, scores

def profile_clusters(X, labels, original_df):
    profiles = {}
    for seg in np.unique(labels):
        mask = labels == seg
        prof = {
            "size": int(mask.sum()),
            "pct": round(mask.sum() / len(labels) * 100, 1),
            "segment_name": SEGMENT_NAMES.get(seg, f"Segment_{seg}"),
            "mean_income": round(original_df.loc[mask, 'income'].mean(), 0),
            "mean_credit_score": round(original_df.loc[mask, 'credit_score'].mean(), 0),
            "mean_employment_years": round(original_df.loc[mask, 'employment_years'].mean(), 1),
            "mean_dti": round(original_df.loc[mask, 'debt_to_income'].mean(), 3),
            "mean_loan_count": round(original_df.loc[mask, 'loan_history_count'].mean(), 1),
            "mean_age": round(original_df.loc[mask, 'age'].mean(), 1),
            "own_pct": round(original_df.loc[mask, 'home_ownership'].isin(['own', 'mortgage']).mean() * 100, 1),
            "verified_income_pct": round(original_df.loc[mask, 'verified_income'].mean() * 100, 1),
        }
        profiles[int(seg)] = prof
    return profiles

def run_clustering(X, n_clusters=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow analysis
    inertias = elbow_analysis(X_scaled)
    
    # Silhouette analysis
    optimal_k, silhouette_scores = find_optimal_k(X_scaled)
    
    # Final clustering with target n_clusters (4 as specified)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, labels)
    
    return {
        "labels": labels,
        "scaler": scaler,
        "model": km,
        "silhouette_score": round(sil_score, 4),
        "elbow_inertias": [round(i, 1) for i in inertias],
        "silhouette_by_k": {str(k): round(s, 4) for k, s in silhouette_scores},
        "optimal_k": optimal_k
    }

if __name__ == "__main__":
    from data_loader import generate_customers
    from features import build_features
    df = generate_customers()
    X = build_features(df)
    result = run_clustering(X)
    print("Silhouette:", result["silhouette_score"])
    print("Optimal K:", result["optimal_k"])