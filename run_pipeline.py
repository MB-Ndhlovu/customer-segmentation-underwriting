import json
import os
import numpy as np
import pandas as pd
from src.data_loader import generate_customer_data, SEGMENT_NAMES
from src.features import build_feature_matrix
from src.segment import find_optimal_k, segment_customers, profile_segments
from src.classify import train_classifier, feature_importance
from sklearn.metrics import silhouette_score as sil_score


print("=" * 60)
print("Customer Segmentation Pipeline — Underwriting Risk Model")
print("=" * 60)

print("\n[1/5] Generating synthetic customer data (n=5000)...")
df = generate_customer_data(n=5000)
print(f"  Shape: {df.shape}")
print(f"  Segments present: {sorted(df['segment_id'].unique())}")

print("\n[2/5] Engineering features (RFM + behavioral + stability)...")
X_scaled, scaler = build_feature_matrix(df, fit=True)
print(f"  Feature matrix shape: {X_scaled.shape}")

print("\n[3/5] Finding optimal k (2-8) via silhouette analysis...")
optimal_k, inertias, silhouettes = find_optimal_k(X_scaled, k_range=(2, 8))
print(f"  Optimal k = {optimal_k}  (silhouette scores: {[round(s,3) for s in silhouettes]})")

print("\n[4/5] Running KMeans clustering (k=4)...")
labels, km = segment_customers(X_scaled, n_clusters=4)
sil = round(sil_score(X_scaled, labels), 4)
print(f"  Silhouette score: {sil}")
profiles = profile_segments(df, labels)
print(f"\n  Segment Profiles (mean values):")
print(profiles.to_string())

print("\n[5/5] Training RandomForest classifier on cluster labels...")
X_raw = df.copy()
clf, X_test, y_test, train_acc, test_acc = train_classifier(X_raw, labels)
print(f"  Train accuracy: {round(train_acc,4)}")
print(f"  Test accuracy:  {round(test_acc,4)}")
print(f"\n  Feature Importance:")
for feat, score in feature_importance(clf).items():
    print(f"    {feat:20s} {score:.4f}")

label_counts = pd.Series(labels).value_counts().sort_index()
segment_labels = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}
summary = {
    "optimal_k": optimal_k,
    "silhouette_score": sil,
    "n_clusters": 4,
    "train_accuracy": round(train_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "segment_distribution": {segment_labels[k]: int(v) for k, v in label_counts.items()},
    "feature_importance": {k: round(float(v), 4) for k, v in feature_importance(clf).items()},
}

os.makedirs("reports", exist_ok=True)
with open("reports/segmentation_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved to reports/segmentation_results.json")

pipeline_output = (
    f"Silhouette: {sil} | Train acc: {round(train_acc,3)} | Test acc: {round(test_acc,3)} | "
    f"Segments: Mass Market={label_counts.get(0,0)}, Rising Prime={label_counts.get(1,0)}, "
    f"Established Prime={label_counts.get(2,0)}, Subprime High-Risk={label_counts.get(3,0)}"
)
print(f"\nPipeline output: {pipeline_output}")

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)