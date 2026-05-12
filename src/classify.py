"""Train RandomForestClassifier on cluster labels for production segment prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train RandomForest on cluster labels. Returns model, metrics, feature importances."""
    X = df[feature_cols].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    feat_importance = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    }

    return {
        "model": clf,
        "accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feat_importance,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_segment(clf, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Use trained classifier to predict segment labels for new data."""
    X = df[feature_cols].values
    return clf.predict(X)


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns
    from segment import run_clustering

    df = generate_customer_data()
    df = build_features(df)
    feats = get_feature_columns()
    seg_result = run_clustering(df, feats)
    labels = seg_result["named_labels"]

    result = train_classifier(df, feats, labels)
    print(f"Classifier Accuracy: {result['accuracy']}")
    print("\nTop Features:")
    for feat, imp in list(result["feature_importance"].items())[:5]:
        print(f"  {feat}: {imp}")