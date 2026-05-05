"""Train RandomForest classifier to predict segment from application features."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_classifier(X, y, test_size=0.2, random_state=42):
    """Train RandomForest classifier and return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, {
        'accuracy': round(float(accuracy), 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_importance(clf, feature_names):
    """Return feature importance as dict."""
    importance = clf.feature_importances_
    return {feat: round(float(imp), 4) for feat, imp in zip(feature_names, importance)}


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features, scale_features
    
    df = generate_customer_data()
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income'
    ]
    X = df[feature_cols]
    y = df['segment_label']
    
    clf, metrics = train_classifier(X, y)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Feature importance: {get_feature_importance(clf, feature_cols)}")