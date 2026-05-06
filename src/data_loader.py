"""Synthetic customer dataset generator for loan underwriting segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)


def generate_customer_data(n=5000):
    """Generate synthetic customer records with realistic distributions."""
    segments = {
        0: dict(income_mean=45000, income_std=12000, credit_mean=640, credit_std=60,
                emp_mean=3, emp_std=1.5, dti_mean=0.28, dti_std=0.08, loans_mean=1.5,
                loans_std=1.0, age_mean=28, age_std=5, home_prob=0.20, verified_prob=0.35),
        1: dict(income_mean=72000, income_std=18000, credit_mean=710, credit_std=50,
                emp_mean=6, emp_std=2.0, dti_mean=0.22, dti_std=0.07, loans_mean=2.0,
                loans_std=1.2, age_mean=35, age_std=6, home_prob=0.45, verified_prob=0.60),
        2: dict(income_mean=110000, income_std=30000, credit_mean=770, credit_std=40,
                emp_mean=10, emp_std=3.0, dti_mean=0.18, dti_std=0.05, loans_mean=2.5,
                loans_std=1.5, age_mean=42, age_std=8, home_prob=0.80, verified_prob=0.85),
        3: dict(income_mean=32000, income_std=8000, credit_mean=560, credit_std=45,
                emp_mean=2, emp_std=1.0, dti_mean=0.40, dti_std=0.10, loans_mean=4.5,
                loans_std=2.0, age_mean=30, age_std=7, home_prob=0.10, verified_prob=0.15),
    }

    records = []
    for seg_id, params in segments.items():
        for _ in range(n // 4):
            income = max(15000, np.random.normal(params['income_mean'], params['income_std']))
            credit_score = max(300, min(850, np.random.normal(params['credit_mean'], params['credit_std'])))
            employment_years = max(0, np.random.normal(params['emp_mean'], params['emp_std']))
            debt_to_income = max(0.05, min(0.80, np.random.normal(params['dti_mean'], params['dti_std'])))
            loan_history_count = max(0, int(np.random.normal(params['loans_mean'], params['loans_std'])))
            age = max(18, min(75, np.random.normal(params['age_mean'], params['age_std'])))
            home_ownership = 1 if np.random.random() < params['home_prob'] else 0
            verified_income = 1 if np.random.random() < params['verified_prob'] else 0

            records.append({
                'income': round(income, 2),
                'credit_score': round(credit_score, 1),
                'employment_years': round(employment_years, 2),
                'debt_to_income': round(debt_to_income, 4),
                'loan_history_count': loan_history_count,
                'age': int(age),
                'home_ownership': home_ownership,
                'verified_income': verified_income,
            })

    df = pd.DataFrame(records)
    return df


def get_feature_names():
    return ['income', 'credit_score', 'employment_years', 'debt_to_income',
            'loan_history_count', 'age', 'home_ownership', 'verified_income']