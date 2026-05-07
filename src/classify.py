from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

FEATURE_COLS = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                'loan_history_count', 'age', 'home_ownership', 'verified_income']

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    report = classification_report(y_test, y_pred, target_names=[
        "Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"
    ], output_dict=True)

    return clf, acc, f1, report

def get_feature_importance(clf):
    return dict(zip(FEATURE_COLS, map(round, clf.feature_importances_, [4]*len(FEATURE_COLS))))