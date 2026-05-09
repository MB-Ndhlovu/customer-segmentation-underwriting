"""Execute full customer segmentation pipeline."""

import json
import sys
import pandas as pd

from src.data_loader import generate_customer_data
from src.features import build_features, FEATURE_COLS
from src.segment import run_segmentation, SEGMENT_NAMES
from src.classify import train_classifier, APP_FEATURES


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    print("\n[1/5] Generating 5000 synthetic customer records...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"      Shape: {df.shape}")
    print(f"      Segments: {df['segment_label'].value_counts().sort_index().to_dict()}")

    print("\n[2/5] Building features (RFM, behavioural, stability)...")
    df = build_features(df)
    print(f"      Total features: {len(FEATURE_COLS)}")

    print("\n[3/5] Running KMeans segmentation (k=4)...")
    seg_result = run_segmentation(df, FEATURE_COLS, n_clusters=4)
    labels = seg_result["cluster_labels"]
    print(f"      Silhouette score: {seg_result['silhouette_at_k4']}")
    print(f"      Elbow inertias: {seg_result['inertias']}")

    print("\n[4/5] Training RandomForestClassifier on cluster labels...")
    clf_result = train_classifier(df, labels)
    print(f"      Test accuracy: {clf_result['accuracy']}")

    print("\n[5/5] Profiling segments...")
    for p in seg_result["profiles"]:
        print(
            f"      [{p['segment_id']}] {p['segment_name']}: "
            f"n={p['count']} ({p['pct']}%) | "
            f"income=${p['income_mean']:,.0f} | "
            f"credit={p['credit_score_mean']} | "
            f"DTI={p['debt_to_income_mean']:.2f} | "
            f"verified={p['verified_income_pct']}%"
        )

    output = {
        "n_records": 5000,
        "n_features": len(FEATURE_COLS),
        "segmentation": {
            "k_used": seg_result["k_used"],
            "silhouette_score": seg_result["silhouette_at_k4"],
            "inertias": seg_result["inertias"],
            "silhouettes": seg_result["silhouettes"],
            "profiles": seg_result["profiles"],
        },
        "classification": {
            "accuracy": clf_result["accuracy"],
            "feature_importances": clf_result["feature_importances"],
        },
    }

    reports_dir = "/home/workspace/Projects/customer-segmentation-underwriting/reports"
    out_path = f"{reports_dir}/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OUTPUT] Saved to {out_path}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(clf_result["classification_report"])

    print("\nFEATURE IMPORTANCES")
    print("-" * 40)
    for feat, imp in sorted(
        clf_result["feature_importances"].items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<30} {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return output


if __name__ == "__main__":
    result = main()