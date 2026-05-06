import numpy as np
import pandas as pd

np.random.seed(42)

def generate_customer_data(n=5000):
    """Generate synthetic customer dataset for underwriting segmentation."""
    segments = []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.35:
            seg = 0  # Mass Market
        elif r < 0.60:
            seg = 1  # Rising Prime
        elif r < 0.82:
            seg = 2  # Established Prime
        else:
            seg = 3  # Subprime High-Risk
        segments.append(seg)

    data = []
    for seg in segments:
        if seg == 0:  # Mass Market
            income = np.random.normal(42000, 12000)
            credit_score = np.random.normal(630, 50)
            employment_years = np.random.exponential(2.5)
            debt_to_income = np.random.normal(0.28, 0.08)
            loan_history_count = np.random.poisson(1.5)
            age = np.random.normal(32, 8)
            home_ownership = np.random.choice([0, 1], p=[0.55, 0.45])
            verified_income = np.random.choice([0, 1], p=[0.40, 0.60])
        elif seg == 1:  # Rising Prime
            income = np.random.normal(68000, 14000)
            credit_score = np.random.normal(700, 45)
            employment_years = np.random.exponential(4.5)
            debt_to_income = np.random.normal(0.22, 0.06)
            loan_history_count = np.random.poisson(2.5)
            age = np.random.normal(38, 7)
            home_ownership = np.random.choice([0, 1], p=[0.35, 0.65])
            verified_income = np.random.choice([0, 1], p=[0.20, 0.80])
        elif seg == 2:  # Established Prime
            income = np.random.normal(105000, 22000)
            credit_score = np.random.normal(760, 40)
            employment_years = np.random.exponential(8.0)
            debt_to_income = np.random.normal(0.18, 0.05)
            loan_history_count = np.random.poisson(3.5)
            age = np.random.normal(45, 8)
            home_ownership = np.random.choice([0, 1], p=[0.10, 0.90])
            verified_income = np.random.choice([0, 1], p=[0.08, 0.92])
        else:  # Subprime High-Risk
            income = np.random.normal(28000, 9000)
            credit_score = np.random.normal(560, 55)
            employment_years = np.random.exponential(1.5)
            debt_to_income = np.random.normal(0.40, 0.10)
            loan_history_count = np.random.poisson(5.0)
            age = np.random.normal(29, 7)
            home_ownership = np.random.choice([0, 1], p=[0.80, 0.20])
            verified_income = np.random.choice([0, 1], p=[0.65, 0.35])

        data.append({
            'income': max(15000, income),
            'credit_score': min(max(500, credit_score), 850),
            'employment_years': max(0, employment_years),
            'debt_to_income': max(0.05, min(debt_to_income, 0.65)),
            'loan_history_count': max(0, loan_history_count),
            'age': min(max(18, age), 70),
            'home_ownership': home_ownership,
            'verified_income': verified_income,
            'segment_label': seg
        })

    return pd.DataFrame(data)