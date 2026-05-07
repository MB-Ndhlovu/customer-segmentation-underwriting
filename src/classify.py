import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_classifier(X, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return clf, acc, report, X_test, y_test, y_pred

def predict_segment(clf, X):
    return clf.predict(X)

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import compute_features
    from segment import cluster
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = compute_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels, _, _ = cluster(X_scaled, n_clusters=4)
    clf, acc, report, _, _, _ = train_classifier(X_scaled, labels)
    print(f"Classifier accuracy: {acc:.4f}")
    print(classification_report(labels, clf.predict(X_scaled), output_dict=True))