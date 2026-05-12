import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_customers
from src.features import build_features
from src.segment import run_clustering, profile_clusters, SEGMENT_NAMES
from src.classify import train_classifier
import joblib

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — UNDERWRITING")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customers()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Home ownership distribution:\n{df['home_ownership'].value_counts().to_string()}")
    
    # Build features
    print("\n[2/5] Engineering features...")
    X = build_features(df)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {list(X.columns)}")
    
    # Clustering
    print("\n[3/5] Running KMeans clustering (k=4)...")
    seg_result = run_clustering(X, n_clusters=4)
    labels = seg_result["labels"]
    print(f"  Silhouette Score: {seg_result['silhouette_score']}")
    print(f"  Optimal K (silhouette): {seg_result['optimal_k']}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Profile clusters
    print("\n[4/5] Profiling segments...")
    profiles = profile_clusters(X, labels, df)
    for seg_id, prof in profiles.items():
        print(f"\n  Segment {seg_id}: {prof['segment_name']}")
        print(f"    Size: {prof['size']} ({prof['pct']}%)")
        print(f"    Avg Income: ${prof['mean_income']:,.0f}")
        print(f"    Avg Credit Score: {prof['mean_credit_score']}")
        print(f"    Avg Employment Years: {prof['mean_employment_years']}")
        print(f"    Avg DTI: {prof['mean_dti']}")
        print(f"    Avg Loan Count: {prof['mean_loan_count']}")
        print(f"    Avg Age: {prof['mean_age']}")
        print(f"    Homeowners: {prof['own_pct']}%")
        print(f"    Verified Income: {prof['verified_income_pct']}%")
    
    # Train classifier
    print("\n[5/5] Training RandomForest classifier...")
    clf_result = train_classifier(X, labels)
    print(f"  Accuracy: {clf_result['accuracy']}")
    print(f"  F1 (weighted): {clf_result['f1_weighted']}")
    print(f"  Train size: {clf_result['train_size']}, Test size: {clf_result['test_size']}")
    print(f"\n  Top 5 Feature Importances:")
    for i, (feat, imp) in enumerate(list(clf_result['feature_importance'].items())[:5], 1):
        print(f"    {i}. {feat}: {imp}")
    
    # Save model
    joblib.dump(clf_result['model'], "models/segment_classifier.pkl")
    print("\n  Model saved: models/segment_classifier.pkl")
    
    # Build results report
    report = {
        "n_customers": int(len(df)),
        "n_features": int(X.shape[1]),
        "silhouette_score": seg_result["silhouette_score"],
        "optimal_k": seg_result["optimal_k"],
        "elbow_inertias": seg_result["elbow_inertias"],
        "silhouette_by_k": seg_result["silhouette_by_k"],
        "segment_profiles": profiles,
        "classifier_accuracy": clf_result["accuracy"],
        "classifier_f1": clf_result["f1_weighted"],
        "classifier_report": clf_result["classification_report"],
        "feature_importance": clf_result["feature_importance"],
        "segment_names": {str(k): v for k, v in SEGMENT_NAMES.items()}
    }
    
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  Report saved: reports/segmentation_results.json")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    # Output summary string for Telegram
    summary = (
        f"Customer Segmentation Pipeline Complete\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Customers: 5000 | Features: {X.shape[1]} | Clusters: 4\n"
        f"Silhouette Score: {seg_result['silhouette_score']} | Classifier Accuracy: {clf_result['accuracy']}\n"
        f"F1 (weighted): {clf_result['f1_weighted']}\n\n"
        f"Segments:\n"
    )
    for seg_id, prof in sorted(profiles.items()):
        summary += f"  {seg_id}. {prof['segment_name']}: {prof['size']} ({prof['pct']}%) | "
        summary += f"Avg Income ${prof['mean_income']:,.0f} | Credit {prof['mean_credit_score']}\n"
    summary += f"\nTop Features: {list(clf_result['feature_importance'].keys())[:3]}"
    
    print(summary)
    return summary

if __name__ == "__main__":
    import numpy as np
    summary = main()