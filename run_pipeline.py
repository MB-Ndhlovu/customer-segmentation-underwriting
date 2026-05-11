"""Full pipeline: load data → engineer features → cluster → classify → save results."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import generate_customer_data
from src.features import build_features
from src.segment import run_clustering, fit_kmeans
from src.classify import run_classification


def main():
    print("=" * 60)
    print("Customer Segmentation for Underwriting — Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(5000)
    print(f"  Segments (true distribution):\n{df['segment_true'].value_counts().sort_index().to_string()}\n")

    # 2. Engineer features
    print("[2/4] Engineering features (RFM, behavioral, stability)...")
    X = build_features(df)
    print(f"  Feature matrix shape: {X.shape}")

    # 3. Clustering
    print("\n[3/4] Running KMeans clustering + silhouette/elbow analysis...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels_arr, _ = fit_kmeans(X_scaled, n_clusters=4)

    cluster_results = run_clustering(X)
    print(f"  Optimal k: {cluster_results['optimal_k']}")
    print(f"  Silhouette scores by k: {dict(zip(cluster_results['k_tested'], [round(s,4) for s in cluster_results['silhouette_scores']]))}")
    print(f"  Cluster distribution: {cluster_results['cluster_counts']}")

    # 4. Classification — pass actual labels array from kmeans
    print("\n[4/4] Training RandomForest classifier on cluster labels...")
    clf_results = run_classification(X, labels_arr)
    print(f"  Accuracy: {clf_results['accuracy']:.4f}")
    print(f"  F1 Macro: {clf_results['f1_macro']:.4f}")

    # Save results
    os.makedirs("reports", exist_ok=True)
    output = {
        "pipeline": "customer-segmentation-underwriting",
        "n_samples": 5000,
        "n_features": int(X.shape[1]),
        "clustering": {
            "optimal_k": cluster_results["optimal_k"],
            "k_tested": cluster_results["k_tested"],
            "silhouette_scores": cluster_results["silhouette_scores"],
            "cluster_counts": {str(k): v for k, v in cluster_results["cluster_counts"].items()},
            "profiles": cluster_results["profiles"],
        },
        "classification": {
            "accuracy": clf_results["accuracy"],
            "f1_macro": clf_results["f1_macro"],
            "feature_importances": clf_results["feature_importances"],
        },
    }

    report_path = "reports/segmentation_results.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[✓] Results saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples:  5000")
    print(f"  Features: {X.shape[1]}")
    print(f"  Clusters: {cluster_results['optimal_k']}")
    print(f"  Silhouette: {max(cluster_results['silhouette_scores']):.4f}")
    print(f"  RF Accuracy: {clf_results['accuracy']:.4f}")
    print(f"  RF F1 Macro: {clf_results['f1_macro']:.4f}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()