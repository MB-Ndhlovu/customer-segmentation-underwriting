"""Train supervised RandomForest classifier on cluster labels."""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import joblib


FEATURE_COLS = [
    "income", "credit_score", "employment_years",
    "debt_to_income", "loan_history_count", "age",
    "home_ownership", "verified_income"
]


def train_classifier(X_raw: pd.DataFrame, labels: np.ndarray):
    """Train RandomForest on cluster labels and return model + metrics."""
    X = X_raw[FEATURE_COLS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=[
        "Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"
    ]))

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature Importances:")
    print(importances.round(4).to_string())

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "confusion_matrix": cm.tolist(),
        "feature_importance": importances.round(4).to_dict(),
    }

    return clf, metrics


def predict_segment(clf, applicant: dict):
    """Predict segment for a single applicant dict."""
    import numpy as np
    vector = np.array([[
        applicant["income"], applicant["credit_score"],
        applicant["employment_years"], applicant["debt_to_income"],
        applicant["loan_history_count"], applicant["age"],
        applicant["home_ownership"], applicant["verified_income"]
    ]])
    label = int(clf.predict(vector)[0])
    return label


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, scale_features
    from segment import fit_kmeans

    df = generate_customer_data()
    X_feat = build_features(df)
    X_scaled, _ = scale_features(X_feat)
    _, labels = fit_kmeans(X_scaled)

    clf, metrics = train_classifier(df, labels)
    print("\nClassifier trained successfully.")