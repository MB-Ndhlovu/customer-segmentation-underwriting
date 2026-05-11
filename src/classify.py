import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(X, labels, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc, classification_report(y_test, y_pred, output_dict=True)


def predict_segment(clf, X):
    return clf.predict(X)