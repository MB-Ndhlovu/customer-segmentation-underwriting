import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def train_classifier(df, feature_cols, target_col='segment_label'):
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk']
        )
    }
    
    return clf, metrics

def get_feature_importance(clf, feature_cols):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    return {
        feature_cols[i]: round(importances[i], 4)
        for i in indices
    }

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import engineer_features, FEATURE_COLS
    from segment import cluster_customers, map_clusters_to_segments
    
    df = generate_customer_data(5000)
    df = engineer_features(df)
    df, km, scaler = cluster_customers(df, FEATURE_COLS)
    df = map_clusters_to_segments(df, FEATURE_COLS)
    
    clf, metrics = train_classifier(df, FEATURE_COLS)
    
    print("Classifier Metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Weighted: {metrics['f1_weighted']}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    importance = get_feature_importance(clf, FEATURE_COLS)
    print("\nTop Feature Importances:")
    for feat, imp in list(importance.items())[:10]:
        print(f"  {feat}: {imp}")