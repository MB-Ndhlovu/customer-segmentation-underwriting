import os
import json
import numpy as np
import pandas as pd

from src.data_loader import generate_customer_data, FEATURE_COLS, SEGMENT_NAMES
from src.features import build_features, scale_features
from src.segment import find_optimal_k, cluster, profile_segments, silhouette_detail
from src.classify import train_classifier


def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/6] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"    Shape: {df.shape}")
    print(f"    Segments embedded: {df['segment_label'].value_counts().sort_index().to_dict()}")

    # ── 2. Feature engineering ───────────────────────────────────
    print("\n[2/6] Building features...")
    X = build_features(df)
    feature_cols = list(X.columns)
    print(f"    Features ({len(feature_cols)}): {feature_cols}")

    X_scaled, scaler = scale_features(X)
    print("    Features scaled (StandardScaler).")

    # ── 3. Clustering ─────────────────────────────────────────────
    print("\n[3/6] Finding optimal K (KMeans)...")
    best_k, best_sil, inertias, silhouettes = find_optimal_k(
        X_scaled, k_range=range(2, 8)
    )
    print(f"    Optimal K (statistical): {best_k}  |  Silhouette: {best_sil:.4f}")
    print(f"    Silhouettes by K: {dict(zip(range(2,8), [f'{s:.4f}' for s in silhouettes]))}")

    # Business requirement: exactly 4 segments
    n_clusters = 4
    print(f"\n[3/6] Clustering with K={n_clusters} (business requirement)...")
    labels = cluster(X_scaled, n_clusters=n_clusters)
    sil_detail = silhouette_detail(X_scaled, labels)
    print(f"    Silhouette detail — mean: {sil_detail['mean']:.4f}, "
          f"std: {sil_detail['std']:.4f}, min: {sil_detail['min']:.4f}, max: {sil_detail['max']:.4f}")

    # ── 4. Segment profiling ─────────────────────────────────────
    print("\n[4/6] Profiling segments...")
    # Use original (non-scaled) features for interpretability
    df["cluster"] = labels
    profiles = profile_segments(df, labels, FEATURE_COLS)
    print("\n  Segment Means:")
    print(profiles.to_string())

    # Map cluster IDs to segment names using centroids similarity
    # We use income + credit_score as the primary discriminator
    seg_cols = ["income", "credit_score", "employment_years", "debt_to_income"]
    centroids = df.groupby("cluster")[seg_cols].mean()
    print("\n  Centroid overview:")
    print(centroids.to_string())

    # Rank centroids by income+credit to assign semantic names
    ranking = centroids["income"].sort_values().index.tolist()
    # lowest income cluster → Subprime High-Risk (3)
    # highest income cluster → Established Prime (2)
    # We remap cluster IDs to semantic names
    cluster_to_semantic = {}
    semantic_order = ["Subprime High-Risk", "Mass Market", "Rising Prime", "Established Prime"]
    for i, cl in enumerate(ranking):
        cluster_to_semantic[cl] = semantic_order[i]

    # Re-encode cluster labels to match business segments 0-3
    # 0=Mass Market, 1=Rising Prime, 2=Established Prime, 3=Subprime High-Risk
    semantic_to_idx = {v: k for k, v in SEGMENT_NAMES.items()}
    labels_mapped = np.array([
        semantic_to_idx[cluster_to_semantic[l]] for l in labels
    ])
    df["segment_label_pred"] = labels_mapped

    # Re-profile with remapped labels
    profiles_mapped = profile_segments(df, labels_mapped, FEATURE_COLS)
    print("\n  Remapped Segment Profiles:")
    print(profiles_mapped.to_string())

    # ── 5. Classification ───────────────────────────────────────
    print("\n[5/6] Training RandomForest classifier on cluster labels...")
    # Use all engineered features for classification
    result = train_classifier(X, labels_mapped)
    print(f"    Test Accuracy: {result['accuracy']:.4f}")
    print("\n  Classification Report:")
    print(classification_report_str(result["classification_report"]))
    print("\n  Confusion Matrix:")
    print(np.array(result["confusion_matrix"]).tolist())
    print("\n  Feature Importances (top 5):")
    imp_series = pd.Series(result["feature_importances"]).sort_values(ascending=False)
    for feat, val in imp_series.head(5).items():
        print(f"    {feat}: {val:.4f}")

    # ── 6. Save results ──────────────────────────────────────────
    print("\n[6/6] Saving artifacts...")
    os.makedirs("reports", exist_ok=True)

    results_json = {
        "optimal_k": int(best_k),
        "silhouette_score": float(sil_detail["mean"]),
        "silhouette_detail": sil_detail,
        "segment_profiles": profiles_mapped.to_dict(),
        "segment_names": SEGMENT_NAMES,
        "classification_accuracy": float(result["accuracy"]),
        "feature_importances": result["feature_importances"],
        "confusion_matrix": result["confusion_matrix"],
        "classification_report": result["classification_report"],
        "n_samples": int(len(df)),
    }
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print("    reports/segmentation_results.json saved.")

    # Summary output
    summary = (
        f"\n{'='*60}\n"
        f"PIPELINE COMPLETE\n"
        f"{'='*60}\n"
        f"  Optimal K:        {best_k}\n"
        f"  Silhouette Score: {sil_detail['mean']:.4f}\n"
        f"  RF Accuracy:      {result['accuracy']:.4f}\n"
        f"  Segments:\n"
    )
    for seg_id, name in sorted(SEGMENT_NAMES.items()):
        count = int((labels_mapped == seg_id).sum())
        pct = count / len(labels_mapped) * 100
        summary += f"    [{seg_id}] {name}: {count} ({pct:.1f}%)\n"

    summary += f"\n  Report: reports/segmentation_results.json\n"
    print(summary)

    return results_json


def classification_report_str(report_dict: dict) -> str:
    lines = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            lines.append(
                f"  {label:>10s}  "
                f"P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  "
                f"F1={metrics['f1-score']:.3f}  "
                f"support={metrics['support']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    os.chdir("/home/workspace/Projects/customer-segmentation-underwriting")
    results = run_pipeline()