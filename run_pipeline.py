import os
import json
import joblib
import numpy as np

from src.data_loader import generate_customer_data
from src.features import engineer_features, FEATURE_COLS
from src.segment import cluster_customers, map_clusters_to_segments, find_optimal_k, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier, get_feature_importance

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)
    
    # Step 1: Load/generate data
    print("\n[1] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(5000)
    print(f"    Shape: {df.shape}")
    print(f"    Segments: {df['segment_name'].value_counts().to_dict()}")
    
    # Step 2: Engineer features
    print("\n[2] Engineering features (RFM, behavioral, stability)...")
    df = engineer_features(df)
    print(f"    Feature columns: {len(FEATURE_COLS)}")
    
    # Step 3: Find optimal K (Elbow + Silhouette)
    print("\n[3] Finding optimal K (Elbow + Silhouette)...")
    ks, inertias, silhouettes = find_optimal_k(df[FEATURE_COLS])
    print(f"    K range tested: {list(ks)}")
    print(f"    Silhouette scores: {[round(s, 3) for s in silhouettes]}")
    best_k_idx = np.argmax(silhouettes)
    best_k = list(ks)[best_k_idx]
    print(f"    Best K by silhouette: {best_k} (score={round(silhouettes[best_k_idx], 3)})")
    
    # Step 4: Cluster customers
    print("\n[4] Clustering customers with KMeans (k=4)...")
    df, km, scaler = cluster_customers(df, FEATURE_COLS, n_clusters=4)
    df = map_clusters_to_segments(df, FEATURE_COLS)
    print(f"    Cluster distribution: {df['segment_label'].value_counts().sort_index().to_dict()}")
    
    # Step 5: Profile segments
    print("\n[5] Profiling segments...")
    profiles = profile_segments(df, FEATURE_COLS)
    for seg_id in sorted(df['segment_label'].unique()):
        seg_df = df[df['segment_label'] == seg_id]
        seg_name = SEGMENT_NAMES[seg_id]
        p = profiles[seg_name]
        print(f"\n    [{seg_name}] {p['count']} customers ({p['pct']}%)")
        print(f"      Income: ${p['features']['income']:,.0f}")
        print(f"      Credit Score: {p['features']['credit_score']}")
        print(f"      DTI: {p['features']['debt_to_income']:.2%}")
        print(f"      Employment Years: {p['features']['employment_years']:.1f}")
    
    # Step 6: Train classifier
    print("\n[6] Training Random Forest classifier...")
    clf, metrics = train_classifier(df, FEATURE_COLS)
    importance = get_feature_importance(clf, FEATURE_COLS)
    print(f"    Accuracy: {metrics['accuracy']:.2%}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    # Save artifacts
    print("\n[7] Saving artifacts...")
    joblib.dump(km, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(clf, 'classifier.pkl')
    print("    Saved: kmeans_model.pkl, scaler.pkl, classifier.pkl")
    
    # Save results JSON
    results = {
        'n_samples': len(df),
        'n_features': len(FEATURE_COLS),
        'feature_columns': FEATURE_COLS,
        'optimal_k': best_k,
        'silhouette_scores': {str(k): round(s, 4) for k, s in zip(ks, silhouettes)},
        'segment_profiles': profiles,
        'classifier_metrics': {
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
        },
        'feature_importance': importance,
    }
    
    with open(os.path.join(REPORTS_DIR, 'segmentation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved: {REPORTS_DIR}/segmentation_results.json")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    results = main()