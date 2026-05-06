"""
Synthetic customer dataset for underwriting segmentation.
Generates 5000 rows with features that naturally cluster into 4 segments.
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def load_customer_data(n=5000):
    # Segment proportions roughly: Mass Market 35%, Rising Prime 30%,
    # Established Prime 20%, Subprime High-Risk 15%
    segment_probs = [0.35, 0.30, 0.20, 0.15]
    labels = np.random.choice(4, size=n, p=segment_probs)

    income, credit_score, employment_years = [], [], []
    debt_to_income, loan_history_count, age = [], [], []
    home_ownership, verified_income = [], []

    for seg in labels:
        if seg == 0:   # Mass Market
            income.append(np.random.normal(52000, 12000))
            credit_score.append(np.random.normal(665, 55))
            employment_years.append(np.random.gamma(4, 1.2))
            debt_to_income.append(np.random.gamma(2.5, 0.8))
            loan_history_count.append(np.random.poisson(2.5))
            age.append(np.random.randint(25, 55))
            home_ownership.append(np.random.choice([0, 1], p=[0.6, 0.4]))
            verified_income.append(np.random.choice([0, 1], p=[0.5, 0.5]))
        elif seg == 1:  # Rising Prime
            income.append(np.random.normal(78000, 15000))
            credit_score.append(np.random.normal(720, 50))
            employment_years.append(np.random.gamma(6, 1.1))
            debt_to_income.append(np.random.gamma(2.2, 0.7))
            loan_history_count.append(np.random.poisson(1.8))
            age.append(np.random.randint(28, 50))
            home_ownership.append(np.random.choice([0, 1], p=[0.35, 0.65]))
            verified_income.append(np.random.choice([0, 1], p=[0.3, 0.7]))
        elif seg == 2:  # Established Prime
            income.append(np.random.normal(115000, 22000))
            credit_score.append(np.random.normal(780, 40))
            employment_years.append(np.random.gamma(10, 1.0))
            debt_to_income.append(np.random.gamma(1.8, 0.6))
            loan_history_count.append(np.random.poisson(1.2))
            age.append(np.random.randint(35, 60))
            home_ownership.append(np.random.choice([0, 1], p=[0.1, 0.9]))
            verified_income.append(np.random.choice([0, 1], p=[0.15, 0.85]))
        else:           # Subprime High-Risk
            income.append(np.random.normal(34000, 9000))
            credit_score.append(np.random.normal(580, 60))
            employment_years.append(np.random.gamma(2.5, 0.9))
            debt_to_income.append(np.random.gamma(4.0, 0.9))
            loan_history_count.append(np.random.poisson(4.5))
            age.append(np.random.randint(22, 50))
            home_ownership.append(np.random.choice([0, 1], p=[0.8, 0.2]))
            verified_income.append(np.random.choice([0, 1], p=[0.7, 0.3]))

    df = pd.DataFrame({
        'income': np.clip(income, 15000, 250000),
        'credit_score': np.clip(credit_score, 300, 850),
        'employment_years': np.clip(employment_years, 0, 45),
        'debt_to_income': np.clip(debt_to_income, 0.1, 15),
        'loan_history_count': np.clip(loan_history_count, 0, 15),
        'age': age,
        'home_ownership': home_ownership,
        'verified_income': verified_income,
        'segment_label': labels
    })

    return df


if __name__ == '__main__':
    df = load_customer_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df['segment_label'].value_counts().sort_index())