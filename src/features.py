"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set from raw customer data.

    Returns DataFrame with:
      - Original 8 features
      - RFM features (income + credit score interaction)
      - Behavioral features
      - Stability features
    """

    X = df.copy()

    # ── RFM features ────────────────────────────────────────────────────────────
    X['income_credit_product'] = X['income'] * X['credit_score'] / 1e6
    X['income_per_employment_year'] = X['income'] / (X['employment_years'] + 1)
    X['credit_to_dti_ratio'] = X['credit_score'] / ((X['debt_to_income'] + 0.01) * 100)

    # ── Behavioral features ───────────────────────────────────────────────────
    X['loan_density'] = X['loan_history_count'] / (X['age'] - 17 + 1)  # loans per available year
    X['income_stability_index'] = X['employment_years'] / (X['age'] - 17 + 1)
    X['verified_income_flag'] = X['verified_income']

    # ── Stability features ─────────────────────────────────────────────────────
    X['home_ownership_flag'] = (X['home_ownership'] >= 1).astype(int)  # owns or mortgaged
    X['income_to_age_ratio'] = X['income'] / (X['age'] + 1)
    X['credit_per_age'] = X['credit_score'] / (X['age'] - 17 + 1)
    X['high_dti_flag'] = (X['debt_to_income'] > 0.45).astype(int)
    X['young_borrower_flag'] = (X['age'] < 25).astype(int)
    X[' employment_years_bucket'] = np.clip(X['employment_years'] // 3, 0, 10).astype(int)

    return X


def get_feature_names() -> list:
    """Return list of features used for clustering / classification."""
    return [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income',
        'income_credit_product', 'income_per_employment_year',
        'credit_to_dti_ratio', 'loan_density', 'income_stability_index',
        'verified_income_flag', 'home_ownership_flag', 'income_to_age_ratio',
        'credit_per_age', 'high_dti_flag', 'young_borrower_flag',
        ' employment_years_bucket'
    ]


def scale_features(df: pd.DataFrame) -> tuple:
    """
    Standard-scale features and return (scaled_df, scaler).
    """
    feature_names = get_feature_names()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_names])
    return pd.DataFrame(scaled, columns=feature_names), scaler