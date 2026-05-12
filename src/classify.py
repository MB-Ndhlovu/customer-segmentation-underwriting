import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def train_classifier(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Feature importance
    importances = clf.feature_importances_
    feature_names = X.columns.tolist()
    feat_imp = dict(zip(feature_names, np.round(importances, 4)))
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "model": clf,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "classification_report": report,
        "feature_importance": feat_imp_sorted,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

def predict_segment(clf, X):
    return clf.predict(X)

if __name__ == "__main__":
    from data_loader import generate_customers
    from features import build_features
    from segment import run_clustering
    
    df = generate_customers()
    X = build_features(df)
    result = run_clustering(X)
    labels = result["labels"]
    
    clf_result = train_classifier(X, labels)
    print("Accuracy:", clf_result["accuracy"])
    print("F1:", clf_result["f1_weighted"])