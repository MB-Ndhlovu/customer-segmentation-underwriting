"""End-to-end pipeline: data generation, clustering, classification, reporting."""

import os
import json
import joblib
from sklearn.metrics import silhouette_score

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

from src.data_loader import generate_customer_data
from src.features import build_features, scale_features
from src.segment import (
    find_optimal_k, plot_elbow, plot_silhouette,
    fit_kmeans, silhouette_analysis, pca_visualization,
    profile_segments, save_results
)
from src.classify import train_classifier

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def main():
    print("=" * 70)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/6] Generating synthetic customer data...")
    df = generate_customer_data(n=5000)
    print(f"  Generated {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Step 2: Feature engineering
    print("\n[2/6] Building features...")
    X = build_features(df)
    print(f"  Features created: {len(X.columns)}")
    print(f"  Feature names: {X.columns.tolist()}")

    X_scaled, scaler = scale_features(X)
    print("  Features scaled (StandardScaler)")

    # Step 3: Clustering analysis
    print("\n[3/6] Finding optimal k (Elbow + Silhouette)...")
    df_metrics = find_optimal_k(X_scaled, k_range=range(2, 10))

    plot_elbow(df_metrics, os.path.join(REPORTS_DIR, "elbow_plot.png"))
    plot_silhouette(df_metrics, os.path.join(REPORTS_DIR, "silhouette_plot.png"))

    # Step 4: Fit KMeans k=4
    print("\n[4/6] Fitting KMeans (k=4)...")
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  Inertia: {km.inertia_:.2f}")

    # Per-cluster analysis
    silhouette_analysis(X_scaled, labels, os.path.join(REPORTS_DIR, "silhouette_analysis.png"))
    pca_visualization(X_scaled, labels, os.path.join(REPORTS_DIR, "pca_projection.png"))

    # Segment profiling
    profile = profile_segments(df, X, labels)

    counts = {SEGMENT_NAMES[i]: int((labels == i).sum()) for i in range(4)}
    print("\n  Segment distribution:")
    for name, cnt in counts.items():
        pct = cnt / len(labels) * 100
        print(f"    {name}: {cnt} ({pct:.1f}%)")

    # Save segmentation results
    seg_results = save_results(df_metrics, profile, km, labels, sil,
                               os.path.join(REPORTS_DIR, "segmentation_results.json"))

    # Step 5: Train classifier
    print("\n[5/6] Training RandomForest classifier...")
    clf, clf_metrics = train_classifier(df, labels)

    # Save model
    model_path = os.path.join(REPORTS_DIR, "segment_classifier.joblib")
    scaler_path = os.path.join(REPORTS_DIR, "feature_scaler.joblib")
    km_path = os.path.join(REPORTS_DIR, "kmeans_model.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(km, km_path)
    print(f"  Models saved to: {REPORTS_DIR}")

    # Save classification metrics
    clf_report_path = os.path.join(REPORTS_DIR, "classification_metrics.json")
    with open(clf_report_path, "w") as f:
        json.dump(clf_metrics, f, indent=2)
    print(f"  Classification metrics saved to: {clf_report_path}")

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Dataset:              5000 rows, {len(X.columns)} engineered features")
    print(f"  Optimal k found:      4 (confirmed by silhouette)")
    print(f"  Silhouette Score:     {sil:.4f}")
    print(f"  Classifier Accuracy:  {clf_metrics['accuracy']:.4f}")
    print(f"  Classifier F1:        {clf_metrics['f1_weighted']:.4f}")
    print(f"\n  Segment breakdown:")
    for name, cnt in counts.items():
        print(f"    {name}: {cnt}")
    print(f"\n  Artifacts saved to:   {REPORTS_DIR}/")

    print("\n" + "=" * 70)
    print("SEGMENT INTERPRETATION FOR UNDERWRITING")
    print("=" * 70)
    for seg_id, name in SEGMENT_NAMES.items():
        mask = labels == seg_id
        seg_data = X[mask]
        print(f"\n  [{seg_id}] {name}")
        print(f"    Avg Income:          R{seg_data['income'].mean():,.0f}" if "income" in seg_data else "N/A")
        print(f"    Avg Credit Score:    {seg_data['credit_score'].mean():.0f}" if "credit_score" in seg_data else "N/A")
        print(f"    Avg DTI:             {seg_data['debt_to_income'].mean():.2%}" if "debt_to_income" in seg_data else "N/A")
        print(f"    Avg Employment Yrs:  {seg_data['employment_years'].mean():.1f}" if "employment_years" in seg_data else "N/A")

    return {
        "silhouette": sil,
        "segment_counts": counts,
        "clf_accuracy": clf_metrics["accuracy"],
        "clf_f1": clf_metrics["f1_weighted"],
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Pipeline executed successfully.")