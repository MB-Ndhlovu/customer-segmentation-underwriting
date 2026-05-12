import numpy as np
import pandas as pd

def build_features(df):
    df = df.copy()
    
    # Encode home_ownership
    home_map = {'rent': 0, 'lease': 1, 'own': 2, 'mortgage': 3}
    df['home_ownership_enc'] = df['home_ownership'].map(home_map).fillna(0)
    df['verified_income_enc'] = df['verified_income'].astype(int)
    
    # RFM-style features
    df['income_per_age'] = df['income'] / df['age'].clip(lower=18)
    df['income_credit_ratio'] = df['income'] / df['credit_score'].clip(lower=1)
    
    # Stability features
    df['emp_stability'] = df['employment_years'] * (df['verified_income_enc'] + 1)
    df['credit_per_emp_year'] = df['credit_score'] / df['employment_years'].clip(lower=0.5)
    
    # Behavioral features
    df['loan_density'] = df['loan_history_count'] / df['age'].clip(lower=18)
    df['dti_risk_flag'] = (df['debt_to_income'] > 0.40).astype(int)
    df['credit_risk_flag'] = (df['credit_score'] < 600).astype(int)
    
    # Composite risk score
    df['risk_score'] = (
        (df['debt_to_income'] * 100).clip(0, 50) +
        (1 - df['credit_score'] / 850) * 100 +
        (1 - df['verified_income_enc']) * 20 +
        df['loan_history_count'] * 3
    )
    
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income_enc',
        'income_per_age', 'income_credit_ratio', 'emp_stability',
        'credit_per_emp_year', 'loan_density', 'dti_risk_flag',
        'credit_risk_flag', 'risk_score'
    ]
    
    return df[feature_cols]

if __name__ == "__main__":
    from data_loader import generate_customers
    df = generate_customers()
    X = build_features(df)
    print(X.describe())