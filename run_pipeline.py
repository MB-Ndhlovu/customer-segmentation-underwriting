import sys
import json
import numpy as np
from src.data_loader import load_data
from src.features import build_features
from src.segment import (
    find_optimal_k, elbow_analysis, run_kmeans, silhouette_analysis,
    profile_segments, save_results
)
from src.classify import train_classifier

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data...")
    df = load_data()
    print(f"  -> {len(df)} customers loaded")

    # 2. Build features
    print("\n[2/5] Engineering features...")
    X, feature_cols, scaler = build_features(df)
    print(f"  -> {X.shape[1]} features built")

    # 3. Clustering
    print("\n[3/5] Running KMeans clustering...")
    k_range, inertias, silhouettes = find_optimal_k(X)
    elbow_k = elbow_analysis(k_range, inertias)

    n_clusters = 4
    km, labels = run_kmeans(X, n_clusters=n_clusters)
    silhouette_scores, _ = silhouette_analysis(X, labels)

    print(f"  -> Optimal K (elbow): {elbow_k}")
    print(f"  -> Silhouette Score: {silhouette_scores:.4f}")

    # 4. Profile segments
    print("\n[4/5] Profiling segments...")
    profiles, seg_assignments = profile_segments(df, labels, feature_cols)

    segment_names = ['Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk']
    for seg_id, name in seg_assignments.items():
        prof = profiles[seg_id]
        print(f"  Segment {seg_id} ({name}):")
        print(f"    Count: {prof['count']} ({prof['pct']}%)")
        print(f"    Avg Income: ${prof['mean_income']:,.0f}")
        print(f"    Avg Credit Score: {prof['mean_credit_score']:.0f}")
        print(f"    Avg DTI: {prof['mean_dti']:.3f}")
        print(f"    Avg Loan Count: {prof['mean_loan_count']:.1f}")

    # 5. Train classifier
    print("\n[5/5] Training RandomForest classifier...")
    clf, acc, report = train_classifier(X, labels)
    print(f"  -> Classification Accuracy: {acc:.4f}")
    print(f"  -> Macro F1: {report['macro avg']['f1-score']:.4f}")

    # Save results
    results = save_results(
        profiles, seg_assignments, float(silhouette_scores),
        n_clusters,
        '/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json'
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: reports/segmentation_results.json")
    print(f"\nSegment Distribution:")
    for seg_id, name in seg_assignments.items():
        prof = profiles[seg_id]
        print(f"  {name}: {prof['count']} ({prof['pct']}%)")
    print(f"\nClustering Silhouette Score: {silhouette_scores:.4f}")
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")

    # Summary dict for Telegram
    summary = {
        'customers': len(df),
        'features': X.shape[1],
        'silhouette_score': round(silhouette_scores, 4),
        'classification_accuracy': round(acc, 4),
        'macro_f1': round(report['macro avg']['f1-score'], 4),
        'segments': {seg_assignments[k]: profiles[k] for k in profiles}
    }
    return summary


if __name__ == '__main__':
    summary = main()