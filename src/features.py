import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df):
    """Build RFM, behavioral, and stability features."""

    # Rename column for consistency
    df = df.copy()

    # RFM-inspired features
    # Recency proxy: employment stability (inverse of turnover)
    df['employment_stability'] = df['employment_years'] / (df['age'] - 18 + 1)

    # Frequency: loan history density
    df['loan_frequency'] = df['loan_history_count'] / (df['age'] - 18 + 1)

    # Monetary: income per age-year
    df['income_per_age'] = df['income'] / (df['age'] + 1)

    # Behavioral features
    df['credit_utilization_estimate'] = df['debt_to_income'] * df['income']
    df['credit_per_employment_year'] = df['loan_history_count'] / (df['employment_years'] + 1)
    df['verified_income_flag'] = df['verified_income']

    # Stability features
    df['income_stability'] = df['income'] / df['income_per_age']
    df['home_ownership_encoded'] = df['home_ownership_status']

    # Composite risk indicators
    df['debt_burden'] = df['debt_to_income'] * df['loan_history_count']
    df['credit_strength'] = df['credit_score'] * (1 - df['debt_to_income'])

    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_status', 'verified_income',
        'employment_stability', 'loan_frequency', 'income_per_age',
        'credit_utilization_estimate', 'credit_per_employment_year',
        'verified_income_flag', 'home_ownership_encoded', 'debt_burden', 'credit_strength'
    ]

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, scaler