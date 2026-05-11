"""
Full pipeline orchestrator for customer segmentation project.
Run: python run_pipeline.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, SEGMENT_NAMES
from src.features import build_features, get_feature_columns
from src.segment import run_segmentation, save_results
from src.classify import train_classifier, save_classification_results


def print_banner():
    print("=" * 65)
    print("  Customer Segmentation for Underwriting — Pipeline")
    print("=" * 65)


def print_segment_profiles(mapping, profiles, segment_order):
    print("\n── Segment Profiles ──────────────────────────────────────────")
    for label in segment_order:
        cid = [k for k, v in mapping.items() if v == label][0]
        p = profiles[cid]
        print(f"\n  [{cid}] {label}")
        print(f"    income             : R{p['income']['mean']:>12,.0f}  (σ={p['income']['std']:,.0f})")
        print(f"    credit_score       :  {p['credit_score']['mean']:>6.0f}  (σ={p['credit_score']['std']:.0f})")
        print(f"    employment_years    :  {p['employment_years']['mean']:>6.1f}  (σ={p['employment_years']['std']:.1f})")
        print(f"    debt_to_income      :  {p['debt_to_income']['mean']:>6.3f}  (σ={p['debt_to_income']['std']:.3f})")
        print(f"    loan_history_count  :  {p['loan_history_count']['mean']:>6.1f}  (σ={p['loan_history_count']['std']:.1f})")
        print(f"    age                 :  {p['age']['mean']:>6.1f}  (σ={p['age']['std']:.1f})")


def main():
    print_banner()

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Rows: {len(df)}")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("\n[2] Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns()
    print(f"    Features: {len(feature_cols)}")

    # ── 3. Segmentation ─────────────────────────────────────────────────────
    print("\n[3] Running KMeans segmentation (k=4)...")
    seg_results = run_segmentation(df, feature_cols, n_clusters=4)
    labels = seg_results["labels"]
    mapping = seg_results["mapping"]
    segment_order = seg_results["segment_order"]
    profiles = seg_results["profiles"]

    # Attach labels to df
    df["segment_label"] = labels
    df["segment_name"] = df["segment_label"].map(mapping)

    print_segment_profiles(mapping, profiles, segment_order)

    # Print silhouette scores
    print("\n── Silhouette Scores ─────────────────────────────────────────")
    for k, score in sorted(seg_results["silhouettes"].items()):
        marker = " ◀ selected" if k == 4 else ""
        print(f"    k={k}: {score:.4f}{marker}")

    # ── 4. Classification ──────────────────────────────────────────────────
    print("\n[4] Training RandomForest classifier...")
    clf, acc, report, importances = train_classifier(df, labels)
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Macro F1 : {report['macro avg']['f1-score']:.4f}")

    # Feature importances
    print("\n── Feature Importances ───────────────────────────────────────")
    sorted_fi = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, fi in sorted_fi:
        bar = "█" * int(fi * 50)
        print(f"    {feat:<28}: {fi:.4f} {bar}")

    # ── 5. Save artifacts ───────────────────────────────────────────────────
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    seg_path = reports_dir / "segmentation_results.json"
    save_results(seg_results, str(seg_path))

    clf_path = reports_dir / "classification_results.json"
    save_classification_results(acc, report, importances, str(clf_path))

    # Segment distribution
    dist = df["segment_name"].value_counts()
    print("\n── Segment Distribution ──────────────────────────────────────")
    for name in segment_order:
        cnt = dist.get(name, 0)
        pct = cnt / len(df) * 100
        bar = "▓" * int(pct / 2)
        print(f"    {name:<24}: {cnt:>5} ({pct:5.1f}%) {bar}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Pipeline complete. Artifacts saved:")
    print(f"  → {seg_path}")
    print(f"  → {clf_path}")
    print("=" * 65)

    # Return summary dict for Telegram
    return {
        "n_rows": len(df),
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "segment_mapping": mapping,
        "segment_order": segment_order,
        "segment_distribution": {k: int(v) for k, v in dist.items()},
        "top_features": [f"{k} ({v:.3f})" for k, v in sorted_fi[:4]],
        "silhouette_k4": seg_results["silhouettes"].get(4, None),
    }


if __name__ == "__main__":
    summary = main()