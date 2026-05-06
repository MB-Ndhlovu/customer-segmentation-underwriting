"""
Feature engineering for customer segmentation.
Three categories: RFM, behavioral, stability.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_features(df):
    """
    Add derived features on top of raw columns.
    Returns X (numpy array) and feature names.
    """
    X = df[['income', 'credit_score', 'employment_years',
            'debt_to_income', 'loan_history_count', 'age',
            'home_ownership', 'verified_income']].values

    feature_names = [
        'income', 'credit_score', 'employment_years',
        'debt_to_income', 'loan_history_count', 'age',
        'home_ownership', 'verified_income'
    ]

    # Scale for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_names


def get_segment_names():
    return {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}