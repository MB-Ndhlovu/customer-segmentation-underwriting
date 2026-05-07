"""Execute full customer segmentation pipeline."""
import sys
import os
import json
import joblib

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.data_loader import generate_customer_data
from src.features   import build_features
from src.segment    import run_segmentation
from src.classify   import train_classifier, save_classifier


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load / generate data
    print("\n[1/5] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(n=5000, seed=42)

    # 2. Build features
    print("[2/5] Engineering features...")
    engineered = build_features(df)
    engineered_cols = list(engineered.columns)

    # 3. Segmentation (KMeans) — features come from `engineered` DataFrame
    print("[3/5] Running KMeans clustering...")
    km, labels, scaler, profile = run_segmentation(engineered, engineered_cols, out_dir='reports')

    # 4. Supervised classification on cluster labels
    print("[4/5] Training RandomForest classifier...")
    clf, acc, f1, importance = train_classifier(engineered, labels, engineered_cols)

    # Save artifacts
    save_classifier(clf, 'reports/rf_classifier.joblib')
    joblib.dump(scaler, 'reports/scaler.joblib')
    joblib.dump(km, 'reports/kmeans_model.joblib')

    # 5. Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 60)

    results = json.load(open('reports/segmentation_results.json'))
    print(f"\nSilhouette Score: {results['silhouette_score']}")
    print(f"RandomForest Accuracy: {acc:.4f}")
    print(f"RandomForest F1 (weighted): {f1:.4f}")
    print(f"\nSegment Distribution:")
    for seg in results['segments']:
        print(f"  {seg['segment_name']}: {seg['count']} ({seg['pct']}%)")

    print(f"\nTop 5 predictive features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\nArtifacts saved to /reports:")
    print("  - segmentation_results.json")
    print("  - rf_classifier.joblib")
    print("  - scaler.joblib")
    print("  - kmeans_model.joblib")
    print("  - elbow_silhouette.png")

    print("\n" + "=" * 60)

    return results, acc, f1


if __name__ == '__main__':
    results, acc, f1 = main()