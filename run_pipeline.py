import json
import numpy as np
from src.data_loader import generate_customer_data
from src.features import build_features, get_engineered_columns, scale_features
from src.segment import run_segmentation
from src.classify import train_classifier, feature_importance


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(n=5000)
    print(f"      Rows: {len(df)} | Columns: {list(df.columns)}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features (RFM, behavioral, stability)...")
    df = build_features(df)
    engineered_cols = get_engineered_columns()
    X_raw = df[engineered_cols].fillna(0).values
    print(f"      Features: {engineered_cols}")

    # 3. Segmentation
    print("\n[3/5] Running KMeans clustering...")
    seg_result = run_segmentation(df, engineered_cols)
    labels = seg_result['labels']
    df['segment_label'] = labels

    print(f"      Silhouette Score: {seg_result['silhouette_score']}")
    print(f"      Segment Sizes: {[seg_result['profiles'][i]['size'] for i in range(4)]}")

    for i in range(4):
        p = seg_result['profiles'][i]
        print(f"      Segment {i} ({p['name']}): n={p['size']} ({p['pct']}%) | "
              f"Income=${p['mean_income']:,.0f} | Credit={p['mean_credit_score']} | "
              f"DTI={p['mean_debt_to_income']} | Loans={p['mean_loan_history_count']}")

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    feature_cols_for_clf = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income'
    ]
    X_clf = df[feature_cols_for_clf].fillna(0).values
    clf_result = train_classifier(X_clf, labels)
    print(f"      Accuracy: {clf_result['accuracy']*100:.1f}%")
    print(f"      Train: {clf_result['train_size']} | Test: {clf_result['test_size']}")

    importance = feature_importance(clf_result['model'], feature_cols_for_clf)
    print("      Top features:")
    for name, score in sorted(importance.items(), key=lambda x: -x[1])[:4]:
        print(f"        {name}: {score}")

    # 5. Save results
    print("\n[5/5] Saving results...")
    results = {
        'silhouette_score': seg_result['silhouette_score'],
        'segment_profiles': seg_result['profiles'],
        'classification_accuracy': clf_result['accuracy'],
        'feature_importance': importance,
        'n_clusters': 4,
        'total_customers': int(len(df))
    }

    with open('reports/segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("      Saved: reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()