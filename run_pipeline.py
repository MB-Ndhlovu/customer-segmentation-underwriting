"""End-to-end customer segmentation pipeline for underwriting."""

import json
import pandas as pd
from src.data_loader import generate_customer_data
from src.features import build_features, get_clustering_features
from src.segment import run_segmentation
from src.classify import train_classifier


def run_pipeline(n_samples: int = 5000, n_clusters: int = 4):
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Generating synthetic customer data...")
    df = generate_customer_data(n_samples=n_samples, seed=42)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
    print(f"  → Segments: {df['segment_name'].value_counts().to_dict()}")

    # Step 2: Feature engineering
    print("\n[2/5] Engineering features (RFM, behavioral, stability)...")
    df_feat = build_features(df)
    feature_df = get_clustering_features(df_feat)
    print(f"  → Clustering feature set: {list(feature_df.columns)}")

    # Step 3: KMeans segmentation
    print("\n[3/5] Running KMeans segmentation (k=4)...")
    seg_result = run_segmentation(df, n_clusters=n_clusters)
    labels = seg_result["labels"]
    silhouette = seg_result["silhouette_mean"]
    profile = seg_result["profile"]

    print(f"  → Silhouette Score: {silhouette}")
    print(f"  → Optimal k found: {seg_result['optimal_k']}")
    print(f"\n  Segment Profiles:\n{profile.to_string()}")

    # Step 4: Train classifier
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    clf_result = train_classifier(df, labels)
    acc = clf_result["accuracy"]

    print(f"  → Accuracy: {acc}")
    print(f"  → Train size: {clf_result['train_size']}, Test size: {clf_result['test_size']}")
    print(f"  → Feature Importances: {clf_result['feature_importances']}")

    # Step 5: Save artifacts
    print("\n[5/5] Saving results...")
    results_path = "reports/segmentation_results.json"

    output = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "classifier_accuracy": acc,
        "optimal_k": seg_result["optimal_k"],
        "segment_profiles": profile.reset_index().to_dict(orient="records"),
        "feature_importances": clf_result["feature_importances"],
        "train_size": clf_result["train_size"],
        "test_size": clf_result["test_size"],
        "classification_report": clf_result["classification_report"],
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  → Saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Silhouette Score : {silhouette}")
    print(f"Classifier Accuracy: {acc}")
    print(f"Segments identified:")
    for seg in profile["segment_name"].values:
        print(f"  - {seg}")

    print("\n✅ Pipeline complete.")

    return output


if __name__ == "__main__":
    output = run_pipeline()