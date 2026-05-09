import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(X, y, feature_names, test_size=0.2, random_state=42):
    """
    Train a RandomForestClassifier on cluster labels.
    Returns model, accuracy, and classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        "Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"
    ])

    # Feature importance
    importance = dict(zip(feature_names, clf.feature_importances_.tolist()))

    return clf, acc, report, importance
