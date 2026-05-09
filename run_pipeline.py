import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data_loader import generate_customer_data
from src.features import build_features, get_feature_columns
from src.segment import find_optimal_k, segment_customers, profile_segments
from src.classify import train_classifier

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"  Data shape: {df.shape}")
    print(f"  Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Build features
    print("\n[2/5] Engineering features...")
    X = build_features(df)
    feature_cols = get_feature_columns()
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    # 3. Scale & cluster
    print("\n[3/5] Running KMeans clustering (k=4)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    inertia, silhouettes = find_optimal_k(X_scaled, range(2, 10))
    print("  K | Inertia  | Silhouette")
    for i, k in enumerate(range(2, 10)):
        print(f"  {k}  | {inertia[i]:8.1f} | {silhouettes[i]:.4f}")

    labels, km = segment_customers(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"\n  Selected k=4  →  Silhouette: {sil:.4f}  |  Inertia: {km.inertia_:.1f}")

    # 4. Profile segments
    print("\n[4/5] Profiling segments...")
    profiles = profile_segments(X, labels, feature_cols)
    print(profiles[["segment_id", "segment_name", "count"]].to_string(index=False))

    # Map cluster IDs to business segment names by looking at cluster centroids
    # Identify which cluster is which based on credit_score and income means
    cluster_stats = []
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        cluster_stats.append({
            "cluster": int(cid),
            "mean_income": X.loc[mask, "income"].mean(),
            "mean_credit_score": X.loc[mask, "credit_score"].mean(),
            "mean_dti": X.loc[mask, "debt_to_income"].mean(),
            "count": int(mask.sum()),
        })

    # Sort by credit_score to map to segment names
    cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x["mean_credit_score"])
    cluster_to_name = {}
    name_order = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
    for i, cs in enumerate(cluster_stats_sorted):
        cluster_to_name[cs["cluster"]] = name_order[i]

    print("\n  Cluster → Segment mapping:")
    for cid, name in sorted(cluster_to_name.items()):
        cs = next(c for c in cluster_stats if c["cluster"] == cid)
        print(f"  Cluster {cid} → {name} "
              f"(income={cs['mean_income']:.0f}, credit={cs['mean_credit_score']:.0f}, dti={cs['mean_dti']:.3f})")

    # 5. Train classifier
    print("\n[5/5] Training RandomForestClassifier on cluster labels...")
    clf, acc, report, importance = train_classifier(X, pd.Series(labels), feature_cols)
    print(f"  Classification accuracy (test): {acc:.4f}")

    print("\n  Classification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg"):
            try:
                seg_name = SEGMENT_NAMES.get(int(label), f"Segment {label}")
            except (ValueError, TypeError):
                seg_name = label
            print(f"  {seg_name}: precision={metrics['precision']:.3f}  "
                  f"recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}  "
                  f"support={int(metrics['support'])}")

    print("\n  Feature Importance (top 10):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # Save results
    results = {
        "silhouette_score": round(sil, 4),
        "inertia": round(float(km.inertia_), 1),
        "n_clusters": 4,
        "n_features": len(feature_cols),
        "n_samples": int(len(df)),
        "cluster_to_segment": {str(k): v for k, v in cluster_to_name.items()},
        "classification_accuracy": round(acc, 4),
        "classification_report": {
            label: {k: round(v, 4) if isinstance(v, float) else int(v) if isinstance(v, int) else v
                   for k, v in metrics.items()}
            for label, metrics in report.items()
            if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg")
        },
        "feature_importance": [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
            for _, row in importance.iterrows()
        ],
        "segment_profiles": [
            {**{k: v for k, v in row.items() if "_mean" in k or "_std" in k or k in ["segment_id", "segment_name", "count"]}}
            for _, row in profiles.iterrows()
        ],
        "k_analysis": {
            "k_range": list(range(2, 10)),
            "inertia": [round(float(x), 1) for x in inertia],
            "silhouettes": [round(float(x), 4) for x in silhouettes],
        },
    }

    import os
    os.makedirs("/home/workspace/Projects/customer-segmentation-underwriting/reports", exist_ok=True)
    out_path = "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print("Pipeline complete.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run()
    print("\n\nSummary:")
    print(f"  Silhouette Score: {results['silhouette_score']}")
    print(f"  Classification Accuracy: {results['classification_accuracy']}")
    print(f"  Segments: {', '.join(results['cluster_to_segment'].values())}")