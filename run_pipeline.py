import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.features import build_features, get_feature_columns
from src.segment import fit_kmeans, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier


def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading synthetic customer data (n=5000)...")
    df = load_data()
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df_fe = build_features(df)
    feature_cols = get_feature_columns()
    X = df_fe[feature_cols].values
    print(f"  Features: {feature_cols}")

    # 3. Scale + cluster
    print("\n[3/5] Scaling features and running KMeans (k=4)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    print(f"  Inertia: {km.inertia_:.2f}")

    # Profile clusters
    profiles = profile_segments(df_fe, labels, feature_cols)

    # Map clusters to business names via dominant original segment
    from scipy.stats import mode
    cluster_to_segment = {}
    for c in sorted(profiles.keys()):
        mask = labels == c
        dominant = mode(df["segment_label"][mask], keepdims=False).mode
        cluster_to_segment[c] = SEGMENT_NAMES[dominant]
        profiles[c]["assigned_name"] = cluster_to_segment[c]

    print("\n  Cluster profiles:")
    for c, p in profiles.items():
        print(f"  Cluster {c} ({cluster_to_segment[c]}):")
        print(f"    count={p['count']}, income={p['mean_income']:,.0f}, "
              f"credit={p['mean_credit_score']:.0f}, DTI={p['mean_dti']:.3f}")

    # 4. Train RandomForest classifier
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    y = labels
    clf, acc, report, importance = train_classifier(X, y, feature_cols)
    print(f"  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(report)

    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
    print("  Top 5 predictive features:")
    for fname, imp in top_features:
        print(f"    {fname}: {imp:.4f}")

    # 5. Save results
    print("\n[5/5] Saving results to reports/segmentation_results.json...")
    results = {
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "cluster_profiles": {str(k): v for k, v in profiles.items()},
        "kmeans_inertia": float(km.inertia_),
        "classifier_accuracy": float(acc),
        "classification_report": report,
        "top_features": {k: float(v) for k, v in importance.items()},
        "silhouette_approx": float(
            __import__("sklearn.metrics").metrics.silhouette_score(
                X_scaled, labels
            )
        ),
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nSilhouette Score: {results['silhouette_approx']:.4f}")
    print(f"RandomForest Accuracy: {results['classifier_accuracy']:.4f}")
    print(f"Results saved to reports/segmentation_results.json")

    # Return summary for Telegram message
    return {
        "silhouette": results["silhouette_approx"],
        "accuracy": results["classifier_accuracy"],
        "profiles": profiles,
        "top_features": top_features,
    }


if __name__ == "__main__":
    summary = run_pipeline()
