import pandas as pd
import numpy as np

def build_features(df):
    X = df[['income', 'credit_score', 'employment_years', 'debt_to_income',
            'loan_history_count', 'age', 'home_ownership', 'verified_income']].copy()

    # Stability score: employment tenure relative to age
    X['stability_score'] = X['employment_years'] / (X['age'] - 18).clip(lower=1)

    # Loan intensity: loan count per year of adulthood
    X['loan_intensity'] = X['loan_history_count'] / (X['age'] - 18).clip(lower=1)

    # Income adequacy: income relative to age-based expected income
    expected_income_by_age = 30000 + (X['age'] - 25).clip(lower=0) * 2000
    X['income_adequacy'] = X['income'] / expected_income_by_age.clip(lower=1)

    # Credit utilization proxy (high loan count + high DTI = risk)
    X['credit_pressure'] = X['debt_to_income'] * (1 + X['loan_history_count'] / 5)

    # Affordability score: inverse of DTI
    X['affordability'] = (0.5 - X['debt_to_income']).clip(lower=0)

    # RFM: Recency proxy via employment stability (no explicit recency in data)
    X['recency_proxy'] = X['employment_years'] * X['verified_income']

    # Behavioral: verified + homeownership combo
    X['verified_asset'] = X['verified_income'] * X['home_ownership']

    feature_cols = list(X.columns)
    return X, feature_cols

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler