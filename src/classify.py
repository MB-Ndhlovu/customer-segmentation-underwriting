from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd

def train_classifier(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return clf, acc, report

def save_model(clf, path="models/classifier.pkl"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import compute_features
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler
    df = generate_synthetic_data()
    X = compute_features(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km, labels = fit_kmeans(Xs, 4)
    clf, acc, report = train_classifier(X, labels)
    print(f"Accuracy: {acc:.4f}")
    print(report)