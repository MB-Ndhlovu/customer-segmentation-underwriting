import json
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, "src")

from src.data_loader import load_data, FEATURE_COLS
from src.features import engineer_features, get_feature_columns, scale_features
from src.segment import (
    elbow_method,
    find_best_k,
    fit_kmeans,
    profile_segments,
    SEGMENT_NAMES,
)
from src.classify import train_classifier, get_feature_importance


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    df = load_data(n=5000)
    print(f"  → {len(df)} records loaded")

    # ── 2. Feature engineering ───────────────────────────────────
    print("\n[2/5] Engineering features …")
    df = engineer_features(df)
    feat_cols = get_feature_columns()
    print(f"  → {len(feat_cols)} features engineered")

    # ── 3. Scale + cluster ───────────────────────────────────────
    print("\n[3/5] Scaling features …")
    X_scaled, scaler = scale_features(df)

    print("  Running Elbow method (k=2..8) …")
    inertias, distortions = elbow_method(X_scaled, range(2, 9))
    print(f"  Inertias: {[round(i, 0) for i in inertias]}")

    best_k, sil_scores = find_best_k(X_scaled, range(2, 9))
    print(f"  Silhouette scores: { {k: round(v,3) for k,v in sil_scores.items()} }")
    print(f"  Best k by silhouette → {best_k}")

    print(f"\n  Fitting KMeans (k=4) …")
    km_model, labels, sil_score, sil_samples = fit_kmeans(X_scaled, n_clusters=4)
    print(f"  Silhouette score: {sil_score:.4f}")

    # ── 4. Profile segments ───────────────────────────────────────
    print("\n[4/5] Profiling segments …")
    df["segment_label"] = labels

    # Identify which cluster ID maps to which business segment
    centroids = km_model.cluster_centers_
    # Use feature index positions for income (0) and credit_score (1)
    # Sort clusters by income desc: highest income = Established Prime (segment 2)
    sorted_cluster_ids = np.argsort([c[0] for c in centroids])[::-1]  # income idx
    # Map: highest income cluster → 2 (Established Prime), next → 1 (Rising Prime), etc.
    # We label based on segment characteristics
    seg_map = {sorted_cluster_ids[0]: 2, sorted_cluster_ids[1]: 1, sorted_cluster_ids[2]: 0, sorted_cluster_ids[3]: 3}

    # Rename for reporting
    renamed = np.array([seg_map[l] for l in labels])
    df["segment_label"] = renamed

    profiles = df.groupby("segment_label").agg({
        "income": "mean",
        "credit_score": "mean",
        "employment_years": "mean",
        "debt_to_income": "mean",
        "loan_history_count": "mean",
        "age": "mean",
        "home_ownership": "mean",
        "verified_income": "mean",
    }).round(2)

    seg_counts = df["segment_label"].value_counts().sort_index()
    print("\n  Segment profiles:")
    for seg_id in sorted(seg_counts.index):
        name = SEGMENT_NAMES.get(seg_id, f"Segment {seg_id}")
        print(f"  [{seg_id}] {name}: n={seg_counts[seg_id]}")
        row = profiles.loc[seg_id]
        print(f"       income={row['income']:.0f}  credit={row['credit_score']:.0f}  "
              f"emp_yrs={row['employment_years']:.1f}  DTI={row['debt_to_income']:.3f}")

    # ── 5. Train classifier ──────────────────────────────────────
    print("\n[5/5] Training supervised classifier …")
    X_clf = df[feat_cols].values
    y_clf = df["segment_label"].values
    clf, (X_tr, X_te, y_tr, y_te), clf_metrics = train_classifier(X_clf, y_clf)

    print(f"  Accuracy: {clf_metrics['accuracy']}")
    print(f"  F1 (weighted): {clf_metrics['f1_weighted']}")
    print("\n  Classification Report:")
    print(clf_metrics["classification_report"])

    feat_imp = get_feature_importance(clf, feat_cols)
    print("  Top 5 feature importances:")
    for i, (f, v) in enumerate(list(feat_imp.items())[:5]):
        print(f"    {i+1}. {f}: {v}")

    # ── 6. Save results ───────────────────────────────────────────
    print("\n[6/6] Saving results …")
    results = {
        "n_records": len(df),
        "n_features": len(feat_cols),
        "silhouette_score": round(sil_score, 4),
        "best_k_by_silhouette": best_k,
        "k_used": 4,
        "segment_counts": {str(k): int(v) for k, v in seg_counts.items()},
        "segment_profiles": profiles.to_dict(orient="index"),
        "centroids": centroids.tolist(),
        "classification_accuracy": clf_metrics["accuracy"],
        "classification_f1_weighted": clf_metrics["f1_weighted"],
        "classification_precision_weighted": clf_metrics["precision_weighted"],
        "classification_recall_weighted": clf_metrics["recall_weighted"],
        "confusion_matrix": clf_metrics["confusion_matrix"],
        "feature_importance": feat_imp,
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  → reports/segmentation_results.json saved")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
