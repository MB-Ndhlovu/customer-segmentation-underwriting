"""Train RandomForestClassifier on cluster labels; predict segment from application features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


APP_FEATURES = [
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


def train_classifier(df, labels, test_size=0.2, random_state=42):
    X = df[APP_FEATURES].values
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 4)

    report = classification_report(
        y_test, y_pred, target_names=[SEGMENT_NAMES[i] for i in sorted(SEGMENT_NAMES)]
    )

    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(APP_FEATURES, clf.feature_importances_)
    }

    return {
        "accuracy": acc,
        "classification_report": report,
        "feature_importances": importances,
    }