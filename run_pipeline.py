import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.features import build_features, encode_categorical, compute_rfm_features, compute_behavioral_features, compute_stability_features
from src.segment import run_segmentation
from src.classify import train_classifier

def run_pipeline():
    print("=" * 60)
    print("Customer Segmentation Pipeline — Underwriting")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} records with columns: {list(df.columns)}")
    segment_counts = df['segment_label'].value_counts().sort_index().to_dict()
    print(f"  Segment distribution: {segment_counts}")

    # 2. Build features
    print("\n[2/5] Engineering features...")
    df_enc = encode_categorical(df)
    df_feat = df_enc.copy()
    df_feat = compute_rfm_features(df_feat)
    df_feat = compute_behavioral_features(df_feat)
    df_feat = compute_stability_features(df_feat)
    X, y = build_features(df)

    feature_cols = list(X.columns)
    print(f"  Built {len(feature_cols)} features: {feature_cols}")

    # Add engineered columns to df_feat for profiling
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = X[col]

    # 3. Scale features
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"  Scaled: mean={X_scaled.mean().mean():.4f}, std={X_scaled.std().mean():.4f}")

    # 4. Run segmentation
    print("\n[4/5] Running KMeans segmentation...")
    labels, kmeans_model, seg_results = run_segmentation(X_scaled, df_feat, feature_cols)
    print(f"  Clusters: {seg_results['n_clusters']}")
    print(f"  Silhouette Score: {seg_results['silhouette_score']}")
    print(f"  Optimal K (elbow): {seg_results['optimal_k']}")
    print(f"  Cluster distribution: {seg_results['cluster_counts']}")

    seg_names = seg_results['segment_names']
    for cluster_id, name in seg_names.items():
        print(f"    Cluster {cluster_id} → {name}")

    # 5. Train classifier
    print("\n[5/5] Training RandomForest classifier...")
    clf, clf_results = train_classifier(X, labels)
    print(f"  Train accuracy: {clf_results['train_accuracy']}")
    print(f"  Test accuracy:  {clf_results['test_accuracy']}")
    print(f"  CV mean score:   {clf_results['cv_mean']} ± {clf_results['cv_std']}")

    top_features = sorted(
        clf_results['feature_importances'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print("  Top 5 features by importance:")
    for feat, imp in top_features:
        print(f"    {feat}: {imp}")

    # Build final results
    results = {
        'segmentation': seg_results,
        'classification': clf_results,
        'feature_columns': feature_cols,
        'n_samples': len(df),
    }

    # Save results
    output_path = '/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return results

if __name__ == '__main__':
    run_pipeline()