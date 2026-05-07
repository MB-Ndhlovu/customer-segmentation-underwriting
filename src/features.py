"""Feature engineering for customer segmentation."""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM, behavioral, and stability features."""

    feat = pd.DataFrame(index=df.index)

    # --- RFM-style features ---
    # Recency proxy: inverse of loan count (more loans = more recent activity)
    feat['recency_proxy'] = 1 / (df['loan_history_count'] + 1)

    # Frequency: loan_history_count (already present)
    feat['frequency'] = df['loan_history_count']

    # Monetary: income
    feat['monetary'] = df['income']

    # --- Behavioral features ---
    feat['credit_per_age']       = df['credit_score'] / df['age']
    feat['income_per_age']       = df['income'] / df['age']
    feat['debt_burden']          = df['debt_to_income'] * df['income']
    feat['employment_stability'] = df['employment_years'] / df['age']

    # --- Stability features ---
    # Home ownership score
    home_score = df['home_ownership'].map({'own': 3, 'mortgage': 2, 'rent': 1})
    feat['home_score'] = home_score

    # Verified income bonus
    feat['verified_income_enc'] = df['verified_income'].astype(int)

    # Income stability proxy: employment_years * income
    feat['income_stability'] = df['employment_years'] * np.log1p(df['income'])

    # Credit trajectory proxy: credit_score / (employment_years + 1)
    feat['credit_trajectory'] = df['credit_score'] / (df['employment_years'] + 1)

    return feat


def get_feature_names() -> list:
    """Return list of feature names used for clustering."""
    return [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income_enc',
        'recency_proxy', 'frequency', 'monetary',
        'credit_per_age', 'income_per_age', 'debt_burden', 'employment_stability',
        'home_score', 'income_stability', 'credit_trajectory',
    ]