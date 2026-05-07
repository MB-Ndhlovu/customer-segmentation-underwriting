import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_synthetic_data
from src.features import build_features
from src.segment import cluster, remap_to_segment_names, profile_segments, find_optimal_k, SEGMENT_NAMES
from src.classify import train_classifier, get_feature_importance

def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic data (5000 rows)...")
    df = generate_synthetic_data(n=5000, seed=42)
    print(f"    Data shape: {df.shape}")
    print(f"    Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Feature engineering
    print("\n[2] Building features...")
    X = build_features(df)
    feature_names = X.columns.tolist()
    print(f"    Features: {feature_names}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Clustering
    print("\n[3] Finding optimal K...")
    elbow, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 9))
    best_k = silhouettes.index(max(silhouettes)) + 2
    print(f"    Best K by silhouette: {best_k} (silhouette={max(silhouettes):.4f})")
    print(f"    Silhouette scores: {[round(s,4) for s in silhouettes]}")

    print(f"\n[4] Clustering with K=4 (business requirement)...")
    raw_labels = cluster(X_scaled, n_clusters=4)
    labels = remap_to_segment_names(df, raw_labels)

    # 5. Profiling
    print("\n[5] Segment profiling...")
    profiles = profile_segments(df, labels, SEGMENT_NAMES)

    for seg_id, p in profiles.items():
        print(f"\n    Segment {seg_id}: {p['name']}")
        print(f"      Count: {p['count']} ({p['pct']:.1f}%)")
        print(f"      Avg Income: ${p['mean_income']:,.0f}")
        print(f"      Avg Credit Score: {p['mean_credit_score']:.0f}")
        print(f"      Avg DTI: {p['mean_debt_to_income']:.3f}")
        print(f"      Avg Employment Years: {p['mean_employment_years']:.1f}")
        print(f"      Home Ownership Rate: {p['home_ownership_rate']:.1%}")
        print(f"      Verified Income Rate: {p['verified_income_rate']:.1%}")

    # 6. Classification
    print("\n[6] Training RandomForest classifier...")
    clf, acc, f1, report, X_test, y_test = train_classifier(X_scaled, labels)
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print("\n    Classification Report:")
    print(report)

    importance = get_feature_importance(clf, feature_names)
    print("    Top features:")
    for _, row in importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # 7. Save results
    results = {
        'n_samples': 5000,
        'n_features': len(feature_names),
        'features': feature_names,
        'optimal_k': best_k,
        'silhouette_scores': {f'k={i+2}': round(s, 4) for i, s in enumerate(silhouettes)},
        'best_silhouette': round(max(silhouettes), 4),
        'classification_accuracy': round(acc, 4),
        'classification_f1': round(f1, 4),
        'segment_profiles': profiles,
        'feature_importance': importance.to_dict(orient='records')
    }

    import os
    os.makedirs('reports', exist_ok=True)
    with open('reports/segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[7] Results saved to reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results

if __name__ == "__main__":
    results = run_pipeline()