import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df):
    """Engineer RFM, behavioral, and stability features."""

    # RFM-style features
    df['income_band'] = pd.cut(df['income'], bins=[0, 35000, 55000, 80000, 200000],
                               labels=[0, 1, 2, 3]).astype(float).fillna(0)
    df['credit_band'] = pd.cut(df['credit_score'], bins=[0, 580, 650, 720, 850],
                                labels=[0, 1, 2, 3]).astype(float).fillna(0)

    # Behavioral
    df['loan_per_year'] = df['loan_history_count'] / (df['employment_years'] + 0.5)
    df['dti_risk'] = (df['debt_to_income'] > 0.35).astype(int)
    df['credit_utilization_proxy'] = 1 - (df['credit_score'] - 500) / 350

    # Stability
    df['employment_stability'] = df['employment_years'] / (df['age'] - 18 + 1)
    df['income_stability'] = df['verified_income'] * (1 / (1 + df['loan_per_year']))

    # Composite risk score
    df['risk_score'] = (
        0.30 * (1 - (df['credit_score'] - 500) / 350) +
        0.25 * df['debt_to_income'] +
        0.20 * df['loan_history_count'] / 10 +
        0.15 * (1 - df['employment_stability']) +
        0.10 * (1 - df['home_ownership'])
    )

    return df


def get_feature_columns():
    """Return feature columns used for clustering."""
    return [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income'
    ]


def get_engineered_columns():
    """Return engineered feature columns used for clustering."""
    return [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income',
        'income_band', 'credit_band', 'loan_per_year', 'dti_risk',
        'credit_utilization_proxy', 'employment_stability', 'income_stability', 'risk_score'
    ]


def scale_features(df, columns):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    return X, scaler