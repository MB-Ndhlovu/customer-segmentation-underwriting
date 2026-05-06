"""Full pipeline: generate data → segment → classify → save results."""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import generate_customer_data, get_feature_columns
from src.segment import run_segmentation, SEGMENT_NAMES
from src.classify import train_segment_classifier


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION — UNDERWRITING PIPELINE")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/4] Generating 5000 synthetic customer records...")
    df = generate_customer_data(n=5000)
    feature_cols = get_feature_columns()
    print(f"    Features: {feature_cols}")
    print(f"    Shape: {df.shape}")

    # Step 2: KMeans segmentation
    print("\n[2/4] Running KMeans segmentation...")
    km, labels, scaler, seg_summary, sil, sil_samples = run_segmentation(
        df, feature_cols, n_clusters=4
    )
    df["segment_label"] = labels
    print(f"    Silhouette Score: {sil:.4f}")

    # Step 3: Train supervised classifier
    print("\n[3/4] Training RandomForest classifier...")
    clf, clf_metrics = train_segment_classifier(
        df, feature_cols, labels, test_size=0.2, random_state=42
    )

    # Step 4: Build final report
    print("\n[4/4] Saving results...")

    # Segment mapping
    seg_name_map = {str(k): v for k, v in SEGMENT_NAMES.items()}

    results = {
        "pipeline": "Customer Segmentation for Underwriting",
        "n_customers": int(len(df)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "segmentation": {
            "method": "KMeans",
            "n_clusters": seg_summary["n_clusters"],
            "silhouette_score": seg_summary["silhouette_score"],
            "k_search_summary": seg_summary["k_search_results"],
            "segment_names": seg_name_map,
            "segment_profiles": seg_summary["segment_profiles"],
        },
        "classification": {
            "method": "RandomForestClassifier",
            "accuracy": clf_metrics["accuracy"],
            "f1_weighted": clf_metrics["f1_weighted"],
            "n_train": clf_metrics["n_train"],
            "n_test": clf_metrics["n_test"],
            "feature_importance": clf_metrics["feature_importance"],
            "confusion_matrix": clf_metrics["confusion_matrix"],
            "classification_report": clf_metrics["classification_report"],
        },
    }

    # Save report
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nSilhouette Score: {sil:.4f}")
    print(f"Optimal k chosen: {seg_summary['n_clusters']}")
    print(f"\nSegment distribution:")
    for cid, prof in seg_summary["segment_profiles"].items():
        print(f"  Cluster {cid} [{prof['segment_name']}]: {prof['count']} customers ({prof['pct']}%)")
    print(f"\nClassifier Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"Classifier F1 (weighted): {clf_metrics['f1_weighted']:.4f}")
    print(f"Top feature: {list(clf_metrics['feature_importance'].keys())[0]}")
    print(f"\nResults saved to: reports/segmentation_results.json")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run()