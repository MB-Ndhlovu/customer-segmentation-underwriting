"""Execute full customer segmentation pipeline."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_customer_data
from src.features import build_features, get_feature_columns
from src.segment import run_clustering, SEGMENT_NAMES
from src.classify import train_classifier


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — Underwriting")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # Step 2: Feature engineering
    print("\n[2/4] Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns()
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Step 3: Clustering
    print("\n[3/4] Running KMeans clustering (k=4)...")
    seg_result = run_clustering(df, feature_cols, n_clusters=4)
    labels = seg_result["named_labels"]
    df["predicted_segment"] = labels

    print(f"  Silhouette Score: {seg_result['silhouette_score']}")
    print("\n  Segment Profiles:")
    for seg_id_str, prof in seg_result["profiles"].items():
        seg_id = int(seg_id_str)
        print(f"  Cluster {seg_id} → {prof['name']}")
        print(f"    Count: {prof['count']} ({prof['pct']}%)")
        print(f"    Avg Income: {prof['means'].get('income', 'N/A'):.2f}")
        print(f"    Avg Credit Score: {prof['means'].get('credit_score', 'N/A'):.0f}")
        print(f"    Avg DTI: {prof['means'].get('debt_to_income', 'N/A'):.4f}")
        print(f"    Verified Income %: {prof['means'].get('verified_income', 'N/A'):.2%}")
        print()

    # Step 4: Classification
    print("[4/4] Training RandomForestClassifier on cluster labels...")
    clf_result = train_classifier(df, feature_cols, labels)
    print(f"  Accuracy: {clf_result['accuracy']}")
    print("\n  Classification Report:")
    for label_idx in sorted(clf_result["classification_report"].keys()):
        if label_idx.isdigit():
            label_name = SEGMENT_NAMES.get(int(label_idx), label_idx)
            metrics = clf_result["classification_report"][label_idx]
            print(
                f"  {label_idx} ({label_name}): "
                f"precision={metrics['precision']:.2f}, "
                f"recall={metrics['recall']:.2f}, "
                f"f1={metrics['f1-score']:.2f}"
            )
    print("\n  Top 5 Feature Importances:")
    for i, (feat, imp) in enumerate(list(clf_result["feature_importance"].items())[:5], 1):
        print(f"    {i}. {feat}: {imp}")

    # Save results
    os.makedirs("reports", exist_ok=True)
    results_summary = {
        "n_customers": int(len(df)),
        "n_features": len(feature_cols),
        "silhouette_score": seg_result["silhouette_score"],
        "cluster_inertias": seg_result["inertias"],
        "cluster_silhouettes": seg_result["silhouettes"],
        "k_values": seg_result["k_values"],
        "classifier_accuracy": clf_result["accuracy"],
        "confusion_matrix": clf_result["confusion_matrix"],
        "feature_importance": clf_result["feature_importance"],
        "segment_profiles": seg_result["profiles"],
        "segment_names": SEGMENT_NAMES,
    }
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\nResults saved to reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Return summary for Telegram message
    return {
        "silhouette_score": seg_result["silhouette_score"],
        "classifier_accuracy": clf_result["accuracy"],
        "profiles": seg_result["profiles"],
        "top_features": dict(list(clf_result["feature_importance"].items())[:3]),
    }


if __name__ == "__main__":
    summary = run()
    # Print as JSON for easy extraction
    print("\n[SUMMARY_JSON]")
    print(json.dumps(summary, indent=2, default=str))