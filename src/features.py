import pandas as pd
import numpy as np

def compute_features(df):
    """
    Build RFM, behavioral, and stability features from raw customer data.

    RFM-style analog:
      - Recency proxy: employment_years (tenure depth)
      - Frequency proxy: loan_history_count
      - Monetary proxy: income / (debt_to_income + 0.01)

    Behavioral:
      - income_credit_ratio
      - dti_stability (inverse DTI risk)

    Stability:
      - employment_to_age_ratio
      - verified_income + home_ownership combo flag
    """
    X = df[['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_history_count', 'age',
            'home_ownership', 'verified_income']].copy()

    X['monetary_proxy'] = X['income'] / (X['debt_to_income'] + 0.01)
    X['income_credit_ratio'] = X['income'] / (X['credit_score'] + 1)
    X['dti_risk'] = X['debt_to_income']  # higher = riskier
    X['tenure_depth'] = X['employment_years'] / (X['age'] - 18 + 1)
    X['stability_score'] = (X['home_ownership'] + X['verified_income'] +
                            (X['employment_years'] > 5).astype(int))

    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income',
        'monetary_proxy', 'income_credit_ratio', 'dti_risk',
        'tenure_depth', 'stability_score'
    ]
    return X[feature_cols]

if __name__ == '__main__':
    from data_loader import generate_customer_data
    df = generate_customer_data()
    X = compute_features(df)
    print(X.describe())