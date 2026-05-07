import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_feature_matrix
from src.segment import find_optimal_k, fit_kmeans, profile_segments, save_results, SEGMENT_NAMES
from src.classify import train_classifier, get_feature_importance

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data...")
    df = generate_customer_data(n=5000)
    print(f"  -> {len(df)} rows generated")
    print(f"  -> Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Features
    print("\n[2/5] Engineering features...")
    X_all, feature_names = build_feature_matrix(df)
    print(f"  -> {X_all.shape[1]} features built")

    # Use only the 8 core features for clustering and classification
    X_core = df[['income', 'credit_score', 'employment_years', 'debt_to_income',
                  'loan_history_count', 'age', 'home_ownership', 'verified_income']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_core)
    print("  -> Core 8 features scaled")

    # 3. KMeans + evaluation
    print("\n[3/5] KMeans clustering with k=4...")
    elbow, silhouettes, best_k = find_optimal_k(X_scaled)
    print(f"  -> Optimal k by silhouette: {best_k}")
    print(f"  -> Silhouette scores: {[round(s,4) for s in silhouettes]}")

    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil_avg = silhouettes[2]  # index 2 = k=4
    print(f"  -> Silhouette @ k=4: {round(sil_avg, 4)}")

    # 4. Profiling
    print("\n[4/5] Profiling segments...")
    profiles = profile_segments(df, labels, feature_names)
    for seg_id, prof in profiles.items():
        print(f"  Segment {seg_id} ({prof['segment_name']}): n={prof['count']} ({prof['pct']}%)")
        print(f"    income={prof['feature_means']['income']}, credit={prof['feature_means']['credit_score']}, "
              f"DTI={prof['feature_means']['debt_to_income']}, age={prof['feature_means']['age']}")

    # 5. Classifier
    print("\n[5/5] Training RandomForest classifier on cluster labels...")
    clf, acc, f1, report = train_classifier(X_core, labels)
    print(f"  -> Accuracy: {round(acc, 4)}")
    print(f"  -> F1 (weighted): {round(f1, 4)}")
    importance = get_feature_importance(clf)
    print(f"  -> Feature importance: {importance}")

    # Save results
    results = save_results(profiles, sil_avg, elbow, silhouettes,
                          "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json")
    results["classifier_accuracy"] = round(acc, 4)
    results["classifier_f1"] = round(f1, 4)
    results["feature_importance"] = importance

    import json
    with open("/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    summary = (
        f"5000 rows | 4 clusters | Silhouette={round(sil_avg,4)} | "
        f"RF Accuracy={round(acc,4)} | F1={round(f1,4)}"
    )
    print(summary)

    return results

if __name__ == "__main__":
    results = main()