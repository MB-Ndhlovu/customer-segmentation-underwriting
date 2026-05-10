"""Train RandomForestClassifier on cluster labels to predict segment from application features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df[feature_cols].values
    y = df["cluster"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Per-class report
    report = classification_report(
        y_test, y_pred_test, target_names=[SEGMENT_NAMES[i] for i in sorted(SEGMENT_NAMES.keys())], output_dict=True
    )

    # Feature importances
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_cols, clf.feature_importances_)
    }

    results = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "classification_report": {k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()},
        "feature_importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return clf, results


def predict_segment(clf, df: pd.DataFrame, feature_cols: list = FEATURE_COLS) -> np.ndarray:
    X = df[feature_cols].values
    return clf.predict(X)