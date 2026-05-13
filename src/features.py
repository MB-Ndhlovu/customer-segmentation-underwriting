import numpy as np
import pandas as pd

def build_features(df):
    """
    Engineer features from raw customer data.
    Three buckets:
      - RFM: income (monetary), employment_years (tenure proxy), loan_history_count (frequency)
      - Behavioral: debt_to_income, verified_income, home_ownership
      - Stability: age, income-to-age ratio, credit_score normal
    """
    X = df[['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_history_count', 'age',
            'home_ownership', 'verified_income']].copy()

    # Derived stability features
    X['income_per_age'] = X['income'] / (X['age'] + 1)
    X['credit_score_norm'] = (X['credit_score'] - 500) / 350  # [0,1] range
    X['employment_stability'] = X['employment_years'] / (X['age'] - 17 + 1)  # fraction of working life
    X['debt_burden'] = X['debt_to_income'] * (1 - X['home_ownership'] * 0.3)  # renters have higher effective burden
    X['income_verified_flag'] = X['verified_income']
    X['loan_density'] = X['loan_history_count'] / (X['employment_years'] + 1)

    return X

if __name__ == '__main__':
    from data_loader import generate_customer_data
    df = generate_customer_data()
    X = build_features(df)
    print(X.describe())