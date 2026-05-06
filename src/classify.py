"""
Train RandomForest classifier on cluster labels.
Simulates production: predict segment from application features alone.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_classifier(X, labels, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=random_state, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== Classification Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        'Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk'
    ]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    print("\nFeature Importances:")
    feat_names = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                  'loan_history_count', 'age', 'home_ownership', 'verified_income']
    for name, imp in sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    return clf, acc, y_test, y_pred


def evaluate_classifier(clf, X, labels):
    y_pred = clf.predict(X)
    return accuracy_score(labels, y_pred)