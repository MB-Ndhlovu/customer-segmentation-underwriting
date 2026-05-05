"""
Full Customer Segmentation Pipeline for Underwriting
"""

import json
import numpy as np
from src.data_loader import generate_synthetic_data, get_feature_matrix, scale_features
from src.features import get_engineered_features
from src.segment import run_segmentation, fit_kmeans, SEGMENT_NAMES
from src.classify import train_segment_classifier, save_model


def print_summary(result: dict, clf_result: dict):
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — SUMMARY")
    print("=" * 60)

    print(f"\n[KMeans Clustering]")
    print(f"  Optimal k (silhouette): {result['optimal_k']}")
    print(f"  Silhouette Score:       {result['silhouette_raw']:.4f}")
    print(f"  Cluster Inertia:        {result['cluster_inertia']:.2f}")

    print(f"\n[Segment Distribution]")
    for name, count in result["segment_counts"].items():
        pct = count / sum(result["segment_counts"].values()) * 100
        print(f"  {name:<25} {count:>5} ({pct:>5.1f}%)")

    print(f"\n[Segment Profiles]")
    for seg, prof in result["segment_profiles"].items():
        print(f"\n  {seg} (n={int(prof['count'])})")
        print(f"    Avg Income:           ${prof['income']:,.0f}")
        print(f"    Avg Credit Score:    {prof['credit_score']:.0f}")
        print(f"    Avg DTI:             {prof['debt_to_income']:.3f}")
        print(f"    Avg Employment Yrs:  {prof['employment_years']:.1f}")
        print(f"    Avg Age:             {prof['age']:.1f}")

    print(f"\n[RandomForest Classifier]")
    print(f"  Test Accuracy:         {clf_result['accuracy']:.4f}")
    print(f"\n  Feature Importances:")
    for feat, imp in sorted(clf_result["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"    {feat:<22} {imp:.4f}")

    print(f"\n  Classification Report:")
    for label, metrics in clf_result["classification_report"].items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            try:
                lbl_int = int(label)
                name = SEGMENT_NAMES.get(lbl_int, label)
            except ValueError:
                name = label
            print(f"    {name:<25} "
                  f"prec={metrics['precision']:.2f}  "
                  f"rec={metrics['recall']:.2f}  "
                  f"f1={metrics['f1-score']:.2f}")

    print("\n" + "=" * 60)


def run():
    # 1. Load data
    print("Loading/generating synthetic data...")
    df = generate_synthetic_data(n_samples=5000, seed=42)
    print(f"  Generated {len(df)} records")

    # 2. Feature engineering
    print("Engineering features...")
    X_raw = get_feature_matrix(df)
    X_eng = get_engineered_features(df)
    print(f"  Raw features: {X_raw.shape[1]}, Engineered: {X_eng.shape[1]}")

    # 3. Scale
    X_scaled, scaler = scale_features(X_raw)

    # 4. Clustering
    print("Running segmentation...")
    seg_result = run_segmentation(X_scaled, df)

    # 5. Relabel using business mapping
    km, raw_labels = fit_kmeans(X_scaled, n_clusters=4)
    from src.segment import map_labels_to_business_segments
    labels = map_labels_to_business_segments(raw_labels, df)

    # 6. Train classifier on raw features → segment labels
    print("\nTraining RandomForest classifier...")
    X_raw_full = get_feature_matrix(df)
    clf_result = train_segment_classifier(X_raw_full, labels)

    # 7. Save artifacts
    print("\nSaving artifacts...")
    save_model(clf_result["model"], "segment_classifier.joblib")
    print("  Saved: segment_classifier.joblib")

    # Save results JSON
    output = {
        "segmentation": {
            "optimal_k": seg_result["optimal_k"],
            "silhouette_score": seg_result["silhouette_raw"],
            "cluster_inertia": seg_result["cluster_inertia"],
            "segment_counts": seg_result["segment_counts"],
            "segment_profiles": seg_result["segment_profiles"]
        },
        "classification": {
            "accuracy": clf_result["accuracy"],
            "feature_importances": clf_result["feature_importances"],
            "test_set_size": int(clf_result["X_test_shape"][0]),
            "y_test_distribution": clf_result["y_test_distribution"]
        }
    }

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Saved: reports/segmentation_results.json")

    # 8. Print summary
    print_summary(seg_result, clf_result)

    print("\nPipeline complete. Artifacts saved to /home/workspace/Projects/customer-segmentation-underwriting/")
    return output


if __name__ == "__main__":
    run()