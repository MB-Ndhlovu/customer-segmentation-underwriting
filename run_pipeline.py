"""Full pipeline orchestrator for customer segmentation underwriting."""

import os
import json
import pandas as pd
import numpy as np

from src.data_loader import load_data, get_feature_columns
from src.features import engineer_features, scale_features
from src.segment import run_segmentation, SEGMENT_NAMES
from src.classify import train_segment_classifier


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n[1/5] Loading synthetic customer data...")
    X_raw, y_raw = load_data(n_rows=5000, seed=42)
    feature_cols = get_feature_columns()
    print(f"  → {len(X_raw)} rows, {len(feature_cols)} features")

    # --- 2. Feature engineering ---
    print("\n[2/5] Engineering features (RFM + behavioral + stability)...")
    X_engineered = engineer_features(X_raw)
    print(f"  → {X_engineered.shape[1]} engineered features: {list(X_engineered.columns)}")

    X_scaled = scale_features(X_engineered)
    print("  → Features standardized")

    # --- 3. Segmentation ---
    print("\n[3/5] Running KMeans segmentation (K=4)...")
    km, labels, seg_results = run_segmentation(
        X_scaled=X_scaled,
        X_orig=X_engineered,
        feature_cols=list(X_engineered.columns),
        n_clusters=4,
        output_dir="reports",
    )
    print(f"  → Silhouette score: {seg_results['silhouette_avg']:.4f}")
    print(f"  → Cluster sizes: {seg_results['cluster_counts']}")

    # --- 4. Classification ---
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    clf_result = train_segment_classifier(X_engineered, pd.Series(labels))
    clf_metrics = clf_result["metrics"]
    print(f"  → Test accuracy: {clf_metrics['test_accuracy']:.4f}")
    print(f"  → CV mean accuracy: {clf_metrics['cv_mean_accuracy']:.4f} ± {clf_metrics['cv_std_accuracy']:.4f}")

    # Feature importance
    print("\n  Top 5 predictive features:")
    for fi in clf_metrics["feature_importance"][:5]:
        print(f"    - {fi['feature']}: {fi['importance']:.4f}")

    # --- 5. Save results ---
    print("\n[5/5] Saving artifacts...")
    os.makedirs("reports", exist_ok=True)

    # Conversion helper
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    seg_profile = seg_results["profile"]
    seg_counts = {
        SEGMENT_NAMES[k]: int(v) for k, v in seg_results["cluster_counts"].items()
    }

    results_summary = {
        "dataset": {
            "n_rows": 5000,
            "n_features": len(feature_cols),
            "n_clusters": 4,
        },
        "segmentation": {
            "silhouette_score": round(float(seg_results["silhouette_avg"]), 4),
            "cluster_counts": seg_counts,
            "profile": {k: {kk: make_serializable(vv) for kk, vv in v.items()} for k, v in seg_profile.items()},
        },
        "classification": {
            "test_accuracy": make_serializable(clf_metrics['test_accuracy']),
            "cv_mean_accuracy": make_serializable(clf_metrics['cv_mean_accuracy']),
            "cv_std_accuracy": make_serializable(clf_metrics['cv_std_accuracy']),
            "feature_importance": clf_metrics["feature_importance"][:8],
        },
    }

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("  → reports/segmentation_results.json")
    print("  → reports/elbow_silhouette.png")

    # --- Summary output ---
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nSilhouette Score:    {seg_results['silhouette_avg']:.4f}")
    print(f"Classifier Accuracy: {clf_metrics['test_accuracy']:.4f}")
    print(f"\nSegment Distribution:")
    for name, count in seg_counts.items():
        print(f"  {name:<25} {count:>5} ({count/50:.1f}%)")
    print(f"\nRepo: https://github.com/MB-Ndhlovu/customer-segmentation-underwriting")

    return results_summary


if __name__ == "__main__":
    results = main()