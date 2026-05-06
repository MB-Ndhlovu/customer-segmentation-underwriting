"""Train RandomForest classifier to predict segment from application features."""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import joblib


def train_segment_classifier(df, feature_cols, labels, test_size=0.2, random_state=42):
    """Train RandomForest to predict segment labels from features."""
    X = df[feature_cols].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== RandomForest Classifier Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = clf.feature_importances_
    feat_imp = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    print("\nTop Feature Importances:")
    for feat, imp in feat_imp:
        print(f"  {feat}: {imp:.4f}")

    return clf, {
        "accuracy": round(float(accuracy), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": {f: round(float(i), 4) for f, i in feat_imp},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


if __name__ == "__main__":
    from data_loader import generate_customer_data, get_feature_columns
    from segment import run_segmentation

    df = generate_customer_data(5000)
    feature_cols = get_feature_columns()
    _, labels, _, _, _, _ = run_segmentation(df, feature_cols)

    clf, metrics = train_segment_classifier(df, feature_cols, labels)
    print("\nTrain/test split results:", metrics["n_train"], "train /", metrics["n_test"], "test")