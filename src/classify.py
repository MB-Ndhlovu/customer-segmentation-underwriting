import numpy as np
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

    feature_importance = dict(zip(
        ['income', 'credit_score', 'employment_years', 'debt_to_income',
         'loan_history_count', 'age', 'home_ownership', 'verified_income',
         'stability_score', 'loan_intensity', 'income_adequacy',
         'credit_pressure', 'affordability', 'recency_proxy', 'verified_asset'],
        clf.feature_importances_.tolist()
    ))

    return {
        'model': clf,
        'accuracy': float(acc),
        'classification_report': report,
        'feature_importance': feature_importance,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }