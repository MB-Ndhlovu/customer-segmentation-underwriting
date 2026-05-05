"""End-to-end pipeline: generate data, segment customers, train classifier, save results."""

import json
import joblib
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_features, scale_features
from src.segment import find_optimal_k, fit_kmeans, profile_segments, map_clusters_to_segments, save_results, SEGMENT_NAMES
from src.classify import train_classifier, get_feature_importance


def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n[1/6] Generating synthetic customer data...")
    df = generate_customer_data(n=5000)
    print(f"  -> Generated {len(df)} records")
    
    # 2. Feature engineering
    print("\n[2/6] Engineering features...")
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income'
    ]
    X_raw = df[feature_cols]
    X_scaled, scaler = scale_features(X_raw)
    print(f"  -> Built {X_scaled.shape[1]} features (scaled)")
    
    # 3. Clustering
    print("\n[3/6] Finding optimal cluster count...")
    optimal_k, inertias, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 8))
    silhouette = silhouettes[optimal_k - 2]
    print(f"  -> Optimal k={optimal_k}, silhouette={silhouette:.4f}")
    
    print("\n[4/6] Fitting KMeans (k=4)...")
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    profiles = profile_segments(df, labels, feature_cols)
    assignment = map_clusters_to_segments(profiles)
    print(f"  -> Clusters assigned to segments")
    
    for cluster_id, seg_id in sorted(assignment.items()):
        p = profiles[cluster_id]
        print(f"  -> Cluster {cluster_id} -> {SEGMENT_NAMES[seg_id]}: "
              f"n={p['size']}, credit={p['credit_score']:.0f}, "
              f"income=R{p['income']:.0f}, DTI={p['debt_to_income']:.2f}")
    
    # 4. Train classifier
    print("\n[5/6] Training RandomForest classifier...")
    clf, clf_metrics = train_classifier(X_raw, df['segment_label'])
    print(f"  -> Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"  -> Train: {clf_metrics['train_size']}, Test: {clf_metrics['test_size']}")
    
    importance = get_feature_importance(clf, feature_cols)
    top_feat = sorted(importance, key=importance.get, reverse=True)[:3]
    print(f"  -> Top features: {', '.join(top_feat)}")
    
    # 5. Save artifacts
    print("\n[6/6] Saving artifacts...")
    results_path = 'reports/segmentation_results.json'
    results = save_results(profiles, silhouette, optimal_k, assignment, results_path)
    
    # Add classifier metrics to results
    results['classifier_accuracy'] = clf_metrics['accuracy']
    results['feature_importance'] = importance
    results['segment_names'] = SEGMENT_NAMES
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    joblib.dump(clf, 'reports/classifier_model.pkl')
    joblib.dump(scaler, 'reports/scaler.pkl')
    print(f"  -> Saved: {results_path}")
    print(f"  -> Saved: reports/classifier_model.pkl")
    print(f"  -> Saved: reports/scaler.pkl")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    summary = (
        f"Segments: 4 | Silhouette: {silhouette:.4f} | "
        f"Classifier Accuracy: {clf_metrics['accuracy']:.4f}"
    )
    print(summary)
    
    return results


if __name__ == '__main__':
    results = run_pipeline()