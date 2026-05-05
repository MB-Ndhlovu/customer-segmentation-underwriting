import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


FEATURE_COLS = ["income", "credit_score", "employment_years", "debt_to_income",
               "loan_history_count", "age", "home_ownership", "verified_income"]


def train_classifier(X_raw, y, test_size=0.2, random_state=42):
    """Train RandomForest on cluster labels using raw application features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw[FEATURE_COLS], y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, X_test, y_test, train_acc, test_acc


def feature_importance(clf):
    """Return feature importance as a sorted Series."""
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    return imp


def predict_segment(clf, applicant_row):
    """Predict segment for a single applicant record."""
    return clf.predict([applicant_row[FEATURE_COLS]])[0]