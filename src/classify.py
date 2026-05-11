import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership", "verified_income",
]

def train_classifier(df, random_state=42):
    X = df[FEATURE_COLS].values
    y = df["segment_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, acc, report

def feature_importance(clf):
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))