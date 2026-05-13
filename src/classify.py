import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_segment_classifier(X, y, test_size=0.2, random_state=42):
    """Train a RandomForest classifier to predict segment from application features"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy, y_test, y_pred

def classify_new_application(clf, application_features):
    """Predict segment for a new application"""
    prediction = clf.predict(application_features)
    probabilities = clf.predict_proba(application_features)
    segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}
    predicted_label = prediction[0]
    return segment_names[predicted_label], probabilities[0]

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_feature_matrix
    from segment import segment_customers

    df = generate_synthetic_data()
    features = build_feature_matrix(df)
    labels, _, _ = segment_customers(features)

    feature_cols = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                    'loan_history_count', 'age', 'home_ownership', 'verified_income']
    X = df[feature_cols]
    y = labels

    clf, accuracy, y_test, y_pred = train_segment_classifier(X, y)

    print(f"Classifier accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))