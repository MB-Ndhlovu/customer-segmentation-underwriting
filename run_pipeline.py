"""Full pipeline: generate data → engineer features → cluster → classify → save results."""

import os
import json
import numpy as np

from src.data_loader import generate_customer_data, get_feature_names
from src.features import compute_rfm_features, compute_behavioral_features, compute_stability_features
from src.segment import find_optimal_k, cluster, profile_segments, save_results
from sklearn.metrics import silhouette_samples
from src.classify import train_classifier, save_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'segmentation_results.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'segment_classifier.joblib')


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(n=5000)
    print(f"    Shape: {df.shape}")
    print(f"    Features: {list(df.columns)}")

    # 2. Engineer features
    print("\n[2] Engineering features...")
    recency, frequency, monetary = compute_rfm_features(df)
    dti_risk, loan_density = compute_behavioral_features(df)
    emp_stability, income_stability, home_bonus = compute_stability_features(df)

    df['recency_score'] = recency
    df['frequency_score'] = frequency
    df['monetary_score'] = monetary
    df['dti_risk'] = dti_risk
    df['loan_density'] = loan_density
    df['emp_stability'] = emp_stability
    df['income_stability'] = income_stability
    df['home_bonus'] = home_bonus

    feature_cols = get_feature_names()
    X = df[feature_cols].values

    # 3. Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Find optimal k
    print("\n[3] Running Elbow + Silhouette analysis (k=2..10)...")
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)
    print(f"    Best k by Silhouette: {best_k}")
    print(f"    Silhouette scores: {[round(s,3) for s in silhouettes]}")

    # For business use-case, enforce 4 segments
    n_clusters = 4
    print(f"    For underwriting business case, using k={n_clusters}")

    # 5. Cluster
    print(f"\n[4] Clustering with KMeans (k={n_clusters})...")
    labels, km_model = cluster(X_scaled, n_clusters=n_clusters)
    df['segment_label'] = labels

    silhouette_avg = np.mean(silhouette_samples(X_scaled, labels))
    print(f"    Silhouette Score: {silhouette_avg:.4f}")

    # 6. Profile segments
    print("\n[5] Profiling segments...")
    profiles, seg_summary = profile_segments(df, labels, feature_cols)
    for seg_id, stats in profiles.items():
        name = seg_summary.get(seg_id, stats['name'])
        print(f"    Segment {seg_id} ({name}): n={stats['count']} ({stats['pct']}%), "
              f"income=${stats['mean_income']:,.0f}, credit={stats['mean_credit_score']:.0f}, "
              f"DTI={stats['mean_debt_to_income']:.2f}")

    # 7. Train classifier
    print("\n[6] Training RandomForest classifier on cluster labels...")
    clf, acc, report = train_classifier(X, labels)
    print(f"    Accuracy: {acc:.4f}")
    print(f"\n    Classification Report:")
    print(report)

    # 8. Save artifacts
    print("\n[7] Saving artifacts...")
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)
    save_results(profiles, seg_summary, silhouette_avg, REPORT_PATH)
    save_model(clf, MODEL_PATH)
    print(f"    Report saved → {REPORT_PATH}")
    print(f"    Model saved  → {MODEL_PATH}")

    # 9. Summary output
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Segments identified:")
    for seg_id in sorted(profiles.keys()):
        name = seg_summary.get(seg_id, profiles[seg_id]['name'])
        print(f"    [{seg_id}] {name}")
    print(f"\n  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Classifier Accuracy: {acc:.4f}")
    print(f"\n  Results: {REPORT_PATH}")

    return {
        'profiles': profiles,
        'seg_summary': seg_summary,
        'silhouette_avg': silhouette_avg,
        'accuracy': acc,
        'report': report,
    }


if __name__ == '__main__':
    results = run()