"""
Supervised classification: predict segment label from application features.
Trains RandomForestClassifier on cluster labels from KMeans.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


APP_FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership_encoded",
    "verified_income",
]


def train_classifier(df: pd.DataFrame, labels: np.ndarray, random_state: int = 42):
    """
    Train a RandomForestClassifier on the cluster labels.
    Returns model, accuracy, report, feature_importances.
    """
    X = df[APP_FEATURE_COLS].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = {
        col: float(fi)
        for col, fi in zip(APP_FEATURE_COLS, clf.feature_importances_)
    }

    return clf, acc, report, importances


def save_classification_results(
    acc: float,
    report: dict,
    importances: dict,
    output_path: str,
):
    payload = {
        "accuracy": acc,
        "classification_report": report,
        "feature_importances": importances,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path