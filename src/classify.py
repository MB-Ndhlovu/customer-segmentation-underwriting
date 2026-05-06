"""Train RandomForest classifier on cluster labels for fast segment prediction."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def train_classifier(df: pd.DataFrame, feature_cols: list, target_col="segment_label", seed=42):
    """Train RandomForest on cluster labels."""
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Per-segment metrics
    report = classification_report(
        y_test, y_pred, target_names=[SEGMENT_NAMES[i] for i in sorted(np.unique(y))], output_dict=True
    )

    results = {
        "accuracy": round(acc, 4),
        "classification_report": {k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()},
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return clf, results


def get_feature_importance(clf, feature_cols: list) -> dict:
    """Return feature importance as sorted dict."""
    importance = dict(zip(feature_cols, clf.feature_importances_))
    importance = {k: round(v, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}
    return importance


def save_artifacts(clf, results, importance, path_prefix="reports/"):
    """Save model and results to disk."""
    import os
    os.makedirs(path_prefix, exist_ok=True)
    joblib.dump(clf, f"{path_prefix}segment_classifier.joblib")
    with open(f"{path_prefix}classification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{path_prefix}feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, scale_features
    from segment import run_segmentation

    df = generate_customer_data()
    df_feat = build_features(df)
    X_scaled, scaler = scale_features(df_feat)
    df_labeled, seg_results, km = run_segmentation(X_scaled, df_feat)

    feature_cols = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
    ]

    clf, results = train_classifier(df_labeled, feature_cols)
    importance = get_feature_importance(clf, feature_cols)

    print(f"Accuracy: {results['accuracy']}")
    print("\nFeature Importance:")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp}")

    save_artifacts(clf, results, importance)