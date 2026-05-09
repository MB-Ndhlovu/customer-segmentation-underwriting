import pandas as pd
import numpy as np

def encode_categorical(df):
    home_map = {'rent': 0, 'own': 1, 'mortgage': 2}
    df = df.copy()
    df['home_ownership_encoded'] = df['home_ownership'].map(home_map)
    return df

def compute_rfm_features(df):
    df = df.copy()
    # Recency proxy: inverse of loan_history (fewer past loans = more recent engagement)
    df['recency_proxy'] = 1 / (df['loan_history_count'] + 1)
    # Frequency: loan_history_count
    df['frequency'] = df['loan_history_count']
    # Monetary: income normalized
    df['monetary'] = df['income'] / 100000
    return df

def compute_behavioral_features(df):
    df = df.copy()
    # Credit utilization proxy via DTI (lower is better)
    df['credit_behavior'] = 1 - df['debt_to_income']
    # Stability score: employment length relative to age
    df['stability_score'] = df['employment_years'] / (df['age'] - 18 + 1)
    # Income per year of employment
    df['income_per_tenure'] = df['income'] / (df['employment_years'] + 1)
    return df

def compute_stability_features(df):
    df = df.copy()
    # Verified income as a stability signal
    df['income_stability'] = df['verified_income']
    # Home ownership as stability indicator (mortgage = most stable)
    df['housing_stability'] = df['home_ownership'].map({'rent': 0, 'own': 1, 'mortgage': 2})
    # Age-adjusted employment tenure
    df['tenure_age_ratio'] = df['employment_years'] / (df['age'] - 18 + 1)
    return df

def build_features(df):
    df = encode_categorical(df)
    df = compute_rfm_features(df)
    df = compute_behavioral_features(df)
    df = compute_stability_features(df)

    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_encoded', 'verified_income',
        'recency_proxy', 'frequency', 'monetary', 'credit_behavior',
        'stability_score', 'income_per_tenure', 'income_stability',
        'housing_stability', 'tenure_age_ratio',
    ]
    return df[feature_cols], df['segment_label']