"""Train RandomForestClassifier on KMeans cluster labels."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership", "verified_income",
]


def train_classifier(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Train RandomForestClassifier on cluster labels; return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURE_COLS], y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Mass Mkt", "Rising Prime", "Est Prime", "Subprime"], output_dict=True)

    importance = dict(zip(FEATURE_COLS, clf.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "model": clf,
        "accuracy": acc,
        "classification_report": report,
        "feature_importance": sorted_imp,
    }


def save_classifier(clf, path: str):
    joblib.dump(clf, path)