"""Feature engineering: RFM, behavioral, and stability features."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw customer data."""
    X = df.copy()

    # Encode home_ownership
    le = LabelEncoder()
    X['home_ownership_enc'] = le.fit_transform(X['home_ownership'])

    # RFM: Recency proxy via loan_history_count + age combo
    X['rfm_recency'] = X['age'] - X['employment_years']
    X['rfm_frequency'] = X['loan_history_count']
    # Monetization proxy: income per employment year
    X['rfm_monetary'] = X['income'] / (X['employment_years'] + 1)

    # Behavioral: debt burden and credit utilization signals
    X['behavioral_dti'] = X['debt_to_income']
    X['behavioral_credit_to_income'] = X['credit_score'] / (X['income'] / 10_000 + 1)
    X['behavioral_loans_per_year'] = X['loan_history_count'] / (X['employment_years'] + 1)

    # Stability: income and employment stability
    X['stability_income_score'] = X['income'] / (X['age'] * 1000 + 1)
    X['stability_tenure_score'] = X['employment_years'] / (X['age'] - 18 + 1)
    X['stability_income_verified'] = X['verified_income'] * X['stability_income_score']

    return X


if __name__ == '__main__':
    from data_loader import generate_customer_data
    df = generate_customer_data()
    feat = build_features(df)
    print(feat.describe())