"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df):
    """Build RFM, behavioral, and stability features from raw customer data."""
    
    # RFM-style features (Recency, Frequency, Monetary adapted for lending)
    df['income_per_employment_year'] = df['income'] / (df['employment_years'] + 1)
    
    # Behavioral features
    df['loan_frequency'] = df['loan_history_count'] / (df['age'] - 18 + 1)
    df['credit_utilization_proxy'] = df['debt_to_income']
    
    # Stability features
    df['employment_stability'] = df['employment_years'] / (df['age'] - 18 + 1)
    df['income_stability'] = df['verified_income']  # Proxy: verified income = more stable
    
    # Risk indicators
    df['high_loan_count'] = (df['loan_history_count'] > 5).astype(int)
    df['high_dti'] = (df['debt_to_income'] > 0.40).astype(int)
    df['low_credit'] = (df['credit_score'] < 600).astype(int)
    
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income',
        'income_per_employment_year', 'loan_frequency', 'employment_stability',
        'verified_income', 'high_loan_count', 'high_dti', 'low_credit'
    ]
    
    return df[feature_cols]


def scale_features(X):
    """Standardize features for clustering."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


if __name__ == '__main__':
    from data_loader import generate_customer_data
    df = generate_customer_data()
    X = build_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(X.describe())