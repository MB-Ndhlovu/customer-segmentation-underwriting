import numpy as np
import pandas as pd

def compute_rfm_features(df):
    df = df.copy()
    
    # Recency proxy: inverse of DTI (lower DTI = more recent/responsible)
    df['payment_behavior_score'] = 1 / (df['debt_to_income'] + 0.1)
    
    # Frequency: loan history density
    df['loan_frequency'] = df['loan_history_count'] / (df['age'] - 17).clip(lower=1)
    
    # Monetary: income per year of employment
    df['income_per_tenure'] = df['income'] / (df['employment_years'].clip(lower=0.5))
    
    return df

def compute_behavioral_features(df):
    df = df.copy()
    
    # Credit utilization proxy
    df['credit_score_norm'] = (df['credit_score'] - 300) / (850 - 300)
    
    # Income stability ratio
    df['income_stability_ratio'] = df['verified_income'] * df['employment_years'] / (df['age'].clip(lower=18) - 17)
    
    # Debt burden score
    df['debt_burden'] = df['debt_to_income'] * df['loan_history_count']
    
    # Home ownership premium (owning = 1, mortgage = 2, renting = 3)
    df['home_ownership_premium'] = (4 - df['home_ownership']) / 3
    
    return df

def compute_stability_features(df):
    df = df.copy()
    
    # Employment stability
    df['employment_stability'] = df['employment_years'] * df['verified_income'] / np.sqrt(df['age'])
    
    # Credit history depth
    df['credit_history_depth'] = (df['age'] - 18) * df['credit_score_norm']
    
    # Loan experience
    df['loan_experience'] = np.where(
        df['loan_history_count'] > 0,
        np.log1p(df['loan_history_count']),
        0
    )
    
    return df

def engineer_features(df):
    df = compute_rfm_features(df)
    df = compute_behavioral_features(df)
    df = compute_stability_features(df)
    return df

FEATURE_COLS = [
    'income', 'credit_score', 'employment_years', 'debt_to_income',
    'loan_history_count', 'age', 'home_ownership', 'verified_income',
    'payment_behavior_score', 'loan_frequency', 'income_per_tenure',
    'credit_score_norm', 'income_stability_ratio', 'debt_burden',
    'home_ownership_premium', 'employment_stability', 'credit_history_depth',
    'loan_experience'
]

if __name__ == '__main__':
    from data_loader import generate_customer_data
    df = generate_customer_data(1000)
    df = engineer_features(df)
    print(df[FEATURE_COLS].describe())