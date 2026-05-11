import numpy as np
import pandas as pd


def build_features(df):
    """
    Build engineered features from raw customer data.
    Returns DataFrame with original + engineered features.
    """
    X = df[['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_history_count', 'age',
            'home_ownership', 'verified_income']].copy()

    # RFM-style features
    X['income_per_employment_year'] = df['income'] / (df['employment_years'] + 0.5)
    X['credit_to_income_ratio'] = df['credit_score'] / (df['income'] / 1000)

    # Behavioral features
    X['loan_density'] = df['loan_history_count'] / (df['age'] - 17 + 1)
    X['credit_per_age'] = df['credit_score'] / (df['age'] - 17 + 1)
    X['income_stability_proxy'] = df['employment_years'] / (df['age'] - 17 + 1)

    # Stability features
    X['debt_burden'] = df['debt_to_income'] * df['loan_history_count']
    X['verified_asset_proxy'] = (df['home_ownership'].astype(int) +
                                  df['verified_income'].astype(int))
    X['employment_depth'] = df['employment_years'] * df['income'] / 100000

    return X


def get_feature_names():
    base = ['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_history_count', 'age',
            'home_ownership', 'verified_income']
    engineered = ['income_per_employment_year', 'credit_to_income_ratio',
                  'loan_density', 'credit_per_age', 'income_stability_proxy',
                  'debt_burden', 'verified_asset_proxy', 'employment_depth']
    return base + engineered