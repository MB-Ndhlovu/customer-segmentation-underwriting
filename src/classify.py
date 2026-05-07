import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk'
}

def train_classifier(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    labels_present = sorted(np.unique(y_test))
    target_names = [SEGMENT_NAMES[l] for l in labels_present]
    report = classification_report(y_test, y_pred, labels=labels_present, target_names=target_names)
    return clf, acc, f1, report, X_test, y_test

def get_feature_importance(clf, feature_names):
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    return imp

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_features
    from segment import cluster, remap_to_segment_names
    from sklearn.preprocessing import StandardScaler

    df = generate_synthetic_data()
    X_feat = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    labels = cluster(X_scaled)
    labels = remap_to_segment_names(df, labels)

    clf, acc, f1, report, _, _ = train_classifier(X_scaled, labels)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print(report)