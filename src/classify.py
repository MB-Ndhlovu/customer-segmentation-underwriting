import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURE_COLS = ["income", "credit_score", "employment_years",
                "debt_to_income", "loan_history_count", "age",
                "home_ownership", "verified_income"]

def train_classifier(X, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        "Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"
    ], output_dict=True)
    return clf, acc, report

if __name__ == "__main__":
    from data_loader import load_data
    from features import build_features
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler

    df = load_data()
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _, labels = fit_kmeans(X_scaled, 4)

    clf, acc, report = train_classifier(df[FEATURE_COLS], labels)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(
        ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"],
        [report[str(i)]["precision"] for i in range(4)],
        target_names=["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
    ))
