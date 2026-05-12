import json
import os
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_features, get_feature_columns
from src.segment import find_optimal_k, fit_kmeans, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000)
    print(f"  Data shape: {df.shape}")

    # 2. Build features
    print("\n[2/5] Engineering features...")
    X = build_features(df)
    feature_cols = get_feature_columns()
    print(f"  Features: {feature_cols}")

    # 3. Scale and cluster
    print("\n[3/5] Finding optimal K (elbow + silhouette)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    k_results = find_optimal_k(X_scaled, k_range=range(2, 8))
    print(f"  K range tested: {k_results['k_range']}")
    print(f"  Silhouette scores: {[round(s, 4) for s in k_results['silhouettes']]}")
    print(f"  Optimal K (max silhouette): {k_results['optimal_k']}")
    print(f"  Best silhouette score: {k_results['best_silhouette']:.4f}")

    # Force 4 clusters for business requirement
    n_clusters = 4
    print(f"\n[3b/5] Fitting KMeans with k={n_clusters}...")
    labels, km = fit_kmeans(X_scaled, n_clusters=n_clusters)
    X["segment_label"] = labels

    # 4. Profile segments
    print("\n[4/5] Profiling segments...")
    profiles = profile_segments(X, labels)

    # Sort segments by credit score (ascending) then DTI (descending) for business clarity
    # Low credit + high DTI = Subprime High-Risk
    # Low credit + low DTI = Mass Market
    # Mid credit + mid DTI = Rising Prime
    # High credit + low DTI = Established Prime
    credit_order = sorted(profiles.keys(), key=lambda k: (profiles[k]["mean_credit_score"], -profiles[k]["mean_debt_to_income"]))

    business_names = ["Subprime High-Risk", "Mass Market", "Rising Prime", "Established Prime"]
    segment_mapping = {}
    for new_id, seg_key in enumerate(credit_order):
        profiles[seg_key]["name"] = business_names[new_id]
        profiles[seg_key]["cluster_id"] = int(seg_key)
        profiles[seg_key]["business_segment_id"] = new_id
        segment_mapping[seg_key] = new_id

    print("\n  Segment Profiles:")
    print("  " + "-" * 56)
    for seg_key in sorted(profiles.keys(), key=lambda k: profiles[k]["mean_income"]):
        p = profiles[seg_key]
        print(f"  {p['name']}: n={p['size']} ({p['pct']}%) | "
              f"Income=${p['mean_income']:,.0f} | "
              f"Credit={p['mean_credit_score']:.0f} | "
              f"DTI={p['mean_debt_to_income']:.2f}")

    # 5. Train classifier
    print("\n[5/5] Training RandomForest classifier...")
    # Map labels to business segment IDs
    business_labels = X["segment_label"].map(lambda x: segment_mapping[x]).values
    clf_result = train_classifier(X, business_labels, feature_cols)
    print(f"  Accuracy: {clf_result['accuracy']:.4f}")
    print("\n  Feature Importances:")
    for feat, imp in sorted(
        clf_result["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"    {feat}: {imp:.4f}")

    # Save results
    os.makedirs("reports", exist_ok=True)
    results = {
        "k_analysis": {
            "k_range": k_results["k_range"],
            "silhouette_scores": k_results["silhouettes"],
            "optimal_k": k_results["optimal_k"],
            "best_silhouette": k_results["best_silhouette"],
        },
        "clustering": {
            "n_clusters": n_clusters,
            "model": "KMeans",
        },
        "segments": profiles,
        "classification": {
            "model": "RandomForestClassifier",
            "accuracy": clf_result["accuracy"],
            "feature_importances": clf_result["feature_importances"],
        },
    }

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: reports/segmentation_results.json")
    print(f"\nSummary:")
    print(f"  - Customers: {len(df)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Silhouette Score: {k_results['best_silhouette']:.4f}")
    print(f"  - Classifier Accuracy: {clf_result['accuracy']:.4f}")

    # Build summary string for Telegram
    summary = (
        f"Pipeline complete:\n"
        f"• 5000 customers segmented into 4 groups\n"
        f"• KMeans silhouette: {k_results['best_silhouette']:.4f}\n"
        f"• RF classifier accuracy: {clf_result['accuracy']:.4f}\n"
        f"• Segments: Mass Market, Rising Prime, Established Prime, Subprime High-Risk"
    )
    return summary


if __name__ == "__main__":
    summary = main()