import json
import sys
import pickle
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from data_loader import generate_synthetic_data
from features import build_features
from segment import run_segmentation
from classify import train_classifier


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # 1. Load / generate data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_synthetic_data(n=5000)
    print(f"  -> Generated {len(df)} records")
    print(f"  -> True segment distribution:\n{df['_true_segment'].value_counts().sort_index().to_dict()}")

    # 2. Feature engineering
    print("\n[2/5] Building features (RFM, behavioral, stability)...")
    df_feat = build_features(df)
    print(f"  -> Total features: {len(df_feat.columns)}")

    # 3. Segmentation
    print("\n[3/5] Running KMeans segmentation (k=4)...")
    df_seg, seg_results = run_segmentation(df_feat)
    print(f"  -> Silhouette score: {seg_results['silhouette_score']:.4f}")
    print(f"  -> Segment counts: {seg_results['segment_counts']}")
    for name, prof in seg_results["profiles"].items():
        print(f"    . {name}: n={prof['n_customers']}  "
              f"avg_income=${prof['income']['mean']:,.0f}  "
              f"avg_credit={prof['credit_score']['mean']:.0f}  "
              f"avg_DTI={prof['debt_to_income']['mean']:.3f}")

    # 4. Classification
    print("\n[4/5] Training RandomForest segment classifier...")
    clf, clf_results = train_classifier(df_seg)
    print(f"  -> Test accuracy: {clf_results['accuracy']:.4f}")
    print(f"  -> CV accuracy:  {clf_results['cv_accuracy_mean']:.4f} +/- {clf_results['cv_accuracy_std']:.4f}")
    print(f"  -> Top features:  {list(clf_results['feature_importance'].keys())[:3]}")

    # 5. Save artifacts
    print("\n[5/5] Saving artifacts...")
    out_dir = pathlib.Path("reports")
    out_dir.mkdir(exist_ok=True)

    # Full results JSON
    output = {
        "pipeline": "customer-segmentation-underwriting",
        "n_customers": int(len(df_seg)),
        "segmentation": {
            "n_clusters": seg_results["n_clusters"],
            "silhouette_score": seg_results["silhouette_score"],
            "segment_counts": seg_results["segment_counts"],
            "profiles": seg_results["profiles"],
        },
        "classification": {
            "accuracy": clf_results["accuracy"],
            "cv_accuracy_mean": clf_results["cv_accuracy_mean"],
            "cv_accuracy_std": clf_results["cv_accuracy_std"],
            "feature_importance": clf_results["feature_importance"],
            "classification_report": clf_results["classification_report"],
            "model_params": clf_results["model_params"],
        },
    }
    results_path = out_dir / "segmentation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  -> {results_path}")

    # Model pickle
    model_path = out_dir / "segment_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  -> {model_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Summary string for Telegram
    summary = (
        f"Pipeline Summary\n"
        f"Customers: {output['n_customers']}\n"
        f"Segments: Mass Market / Rising Prime / Established Prime / Subprime High-Risk\n"
        f"Silhouette Score: {output['segmentation']['silhouette_score']:.4f}\n"
        f"Segment Counts: {output['segmentation']['segment_counts']}\n"
        f"Classifier Accuracy: {output['classification']['accuracy']:.4f}\n"
        f"CV Accuracy: {output['classification']['cv_accuracy_mean']:.4f} +/- {output['classification']['cv_accuracy_std']:.4f}\n"
        f"Top Features: {', '.join(list(output['classification']['feature_importance'].keys())[:3])}"
    )
    print(summary)
    return output


if __name__ == "__main__":
    output = main()