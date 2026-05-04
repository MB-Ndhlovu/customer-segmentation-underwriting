import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership",
    "verified_income",
]

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def train_classifier(df: pd.DataFrame) -> dict:
    """
    Train a RandomForestClassifier on cluster labels so we can predict
    segment from raw application features (without re-running clustering).
    """
    X = df[FEATURE_COLS].values
    y = df["segment_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    # Feature importance
    importance = {
        col: float(imp)
        for col, imp in zip(FEATURE_COLS, clf.feature_importances_)
    }
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    report = classification_report(
        y_test, y_pred, target_names=list(SEGMENT_NAMES.values()), output_dict=True
    )

    results = {
        "accuracy": float(acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "feature_importance": importance_sorted,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_params": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
    }

    return clf, results


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_features
    from segment import run_segmentation

    df = generate_synthetic_data()
    df_feat = build_features(df)
    df_seg, _ = run_segmentation(df_feat)
    clf, results = train_classifier(df_seg)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print("\nFeature Importance:")
    for k, v in results["feature_importance"].items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(json.dumps(results["classification_report"], indent=2))