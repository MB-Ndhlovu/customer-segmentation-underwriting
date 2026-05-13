import json
import numpy as np
from src.data_loader import generate_synthetic_data
from src.features import build_feature_matrix
from src.segment import segment_customers, profile_segments
from src.classify import train_segment_classifier

def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load/Generate Data
    print("\n[1/5] Generating synthetic customer dataset (5000 rows)...")
    df = generate_synthetic_data(n=5000, seed=42)
    print(f"  -> Generated {len(df)} rows")
    print(f"  -> Segments: {dict(df['segment_label'].value_counts().sort_index())}")

    # 2. Build Features
    print("\n[2/5] Computing features (RFM, behavioral, stability)...")
    features = build_feature_matrix(df)
    print(f"  -> Feature matrix shape: {features.shape}")

    # 3. KMeans Clustering
    print("\n[3/5] Running KMeans clustering (k=4)...")
    labels, scaler, kmeans = segment_customers(features, n_clusters=4)
    print(f"  -> Cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 4. Segment Profiling
    print("\n[4/5] Profiling segments...")
    profiles, df_labelled = profile_segments(df, labels)

    segment_names = ['Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk']
    results = {
        'n_samples': int(len(df)),
        'n_features': int(features.shape[1]),
        'cluster_counts': {},
        'segment_profiles': profiles
    }

    for name in segment_names:
        stats = profiles[name]
        results['cluster_counts'][name] = int(stats['count'])
        print(f"\n  {name} ({stats['count']} customers, {stats['pct']:.1f}%):")
        print(f"    Avg Income: ${stats['mean_income']:,.0f}")
        print(f"    Avg Credit Score: {stats['mean_credit_score']:.0f}")
        print(f"    Avg Employment Years: {stats['mean_employment_years']:.2f}")
        print(f"    Avg DTI: {stats['mean_debt_to_income']:.2f}")
        print(f"    Avg Loan History: {stats['mean_loan_history_count']:.1f}")
        print(f"    Avg Age: {stats['mean_age']:.1f}")
        print(f"    Home Owners: {stats['pct_home_owners']:.1f}%")
        print(f"    Verified Income: {stats['pct_verified_income']:.1f}%")

    # 5. Train Supervised Classifier
    print("\n[5/5] Training RandomForest classifier...")
    feature_cols = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                    'loan_history_count', 'age', 'home_ownership', 'verified_income']
    X = df[feature_cols]
    y = labels

    clf, accuracy, y_test, y_pred = train_segment_classifier(X, y, test_size=0.2)

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n  -> Classifier Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=segment_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=segment_names))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"  {segment_names}")
    print(f"  {cm}")

    results['classifier_accuracy'] = float(accuracy)
    results['classification_report'] = report
    results['confusion_matrix'] = cm.tolist()

    # Save results
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    with open('reports/segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to reports/segmentation_results.json")

    return results

if __name__ == "__main__":
    results = run_pipeline()