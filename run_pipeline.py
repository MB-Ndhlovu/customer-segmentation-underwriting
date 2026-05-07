"""End-to-end pipeline: generate data -> engineer features -> segment -> classify -> report."""

import json
import sys

from src.data_loader import load_customer_data
from src.features import build_features, get_feature_names
from src.segment import fit_kmeans, find_optimal_k, profile_segments, assign_label_names, LABEL_NAMES
from src.classify import train_classifier, get_feature_importance
from sklearn.preprocessing import StandardScaler


def run():
    # 1. Load data
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_customer_data(n=5000)
    print(f"Generated {len(df)} customer records")
    print(df.describe().round(2).to_string())
    print()

    # 2. Feature engineering
    print("=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    X = build_features(df)
    feature_names = get_feature_names()
    print(f"Built {len(feature_names)} features: {feature_names}")
    print()

    # 3. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Clustering analysis
    print("=" * 60)
    print("STEP 3: Clustering Analysis (KMeans)")
    print("=" * 60)
    inertias, silhouettes = find_optimal_k(X_scaled, (2, 8))
    print(f"Inertia per k: {dict(zip(range(2,9), [round(i,1) for i in inertias]))}")
    print(f"Silhouette per k: {dict(zip(range(2,9), [round(s,4) for s in silhouettes]))}")
    print(f"Selected k=4 (business requirement)")

    km, raw_labels, centroid_map, label_names = fit_kmeans(X_scaled, k=4)
    segment_labels = assign_label_names(raw_labels, centroid_map)

    silhouette_avg = round(silhouettes[2], 4)  # index 2 = k=4
    print(f"KMeans fitted. Centroid map: {centroid_map}")
    print(f"Segment labels: {LABEL_NAMES}")
    print(f"Silhouette score (k=4): {silhouette_avg}")
    print()

    # 5. Segment profiling
    print("=" * 60)
    print("STEP 4: Segment Profiling")
    print("=" * 60)
    profiles = profile_segments(X, segment_labels)
    print(profiles.round(2).to_string())
    print()

    # 6. Classification
    print("=" * 60)
    print("STEP 5: Supervised Classification (RandomForest)")
    print("=" * 60)
    clf, metrics = train_classifier(X, segment_labels)
    print(f"Test Accuracy: {metrics['test_accuracy']}")
    print(f"CV Mean Accuracy: {metrics['cv_mean_accuracy']} ± {metrics['cv_std']}")
    print()

    importance = get_feature_importance(clf, feature_names)
    print("Top 10 Feature Importances:")
    print(importance.head(10).to_string(index=False))
    print()

    # 7. Build results
    results = {
        "n_customers": int(len(df)),
        "n_features": len(feature_names),
        "k_chosen": 4,
        "silhouette_score": silhouette_avg,
        "inertia": round(float(km.inertia_), 2),
        "clustering_metrics": {
            "inertias": {f"k={k}": round(v, 2) for k, v in zip(range(2, 9), inertias)},
            "silhouettes": {f"k={k}": round(v, 4) for k, v in zip(range(2, 9), silhouettes)},
        },
        "segment_labels": LABEL_NAMES,
        "classification_metrics": metrics,
        "segment_profiles": profiles.round(4).to_dict(),
        "top_features": importance.head(10).to_dict(orient="records"),
    }

    # 8. Save
    out_path = "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Customers:    {results['n_customers']}")
    print(f"Features:     {results['n_features']}")
    print(f"Silhouette:   {silhouette_avg}")
    print(f"CV Accuracy:  {metrics['cv_mean_accuracy']}")
    print(f"Test Acc:     {metrics['test_accuracy']}")

    return results


if __name__ == "__main__":
    results = run()