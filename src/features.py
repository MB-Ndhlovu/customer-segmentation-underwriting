"""Feature engineering: RFM, behavioral, and stability features."""

import numpy as np


def compute_rfm_features(df):
    """RFM-style recency features derived from credit behaviour."""
    recency_score = (df['credit_score'] - 300) / (850 - 300)  # normalized credit health
    frequency_score = 1 / (df['loan_history_count'] + 1)      # lower loans = better
    monetary_score = df['income'] / 100000                      # income relative to benchmark
    return recency_score, frequency_score, monetary_score


def compute_behavioral_features(df):
    """Behavioural indicators: debt burden and loan velocity."""
    dti_risk = (df['debt_to_income'] - 0.10) / 0.60            # normalized DTI risk
    loan_density = df['loan_history_count'] / (df['age'] - 17) # loans per eligible year
    return dti_risk, loan_density


def compute_stability_features(df):
    """Employment and income stability proxies."""
    emp_stability = df['employment_years'] / (df['age'] - 17)  # employment as fraction of working life
    income_stability = df['verified_income']                   # verified = more stable
    home_bonus = df['home_ownership'] * 0.2                    # homeownership adds stability
    return emp_stability, income_stability, home_bonus