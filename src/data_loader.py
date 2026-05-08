"""Synthetic customer dataset generator."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer records across 4 distinct segments."""
    np.random.seed(seed)

    # Segment proportions
    n0 = int(n * 0.35)  # Mass Market
    n1 = int(n * 0.30)  # Rising Prime
    n2 = int(n * 0.20)  # Established Prime
    n3 = n - n0 - n1 - n2  # Subprime High-Risk

    segments = (
        [0] * n0 + [1] * n1 + [2] * n2 + [3] * n3
    )
    np.random.shuffle(segments)

    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership = [''] * n
    verified_income = np.zeros(n)

    for i, seg in enumerate(segments):
        if seg == 0:  # Mass Market
            income[i] = np.random.normal(55_000, 15_000)
            credit_score[i] = np.random.normal(660, 50)
            employment_years[i] = np.random.gamma(4, 2)
            debt_to_income[i] = np.random.beta(4, 5) * 0.4 + 0.1
            loan_history_count[i] = np.random.poisson(2)
            age[i] = np.random.normal(38, 8)
            home_ownership[i] = np.random.choice(['rent', 'own'], p=[0.6, 0.4])
            verified_income[i] = np.random.choice([0, 1], p=[0.5, 0.5])

        elif seg == 1:  # Rising Prime
            income[i] = np.random.normal(80_000, 20_000)
            credit_score[i] = np.random.normal(720, 45)
            employment_years[i] = np.random.gamma(6, 2.5)
            debt_to_income[i] = np.random.beta(3, 6) * 0.35 + 0.05
            loan_history_count[i] = np.random.poisson(3)
            age[i] = np.random.normal(34, 6)
            home_ownership[i] = np.random.choice(['rent', 'own'], p=[0.4, 0.6])
            verified_income[i] = np.random.choice([0, 1], p=[0.2, 0.8])

        elif seg == 2:  # Established Prime
            income[i] = np.random.normal(130_000, 35_000)
            credit_score[i] = np.random.normal(780, 40)
            employment_years[i] = np.random.gamma(10, 2.5)
            debt_to_income[i] = np.random.beta(2, 8) * 0.25 + 0.02
            loan_history_count[i] = np.random.poisson(4)
            age[i] = np.random.normal(45, 8)
            home_ownership[i] = np.random.choice(['rent', 'own'], p=[0.1, 0.9])
            verified_income[i] = np.random.choice([0, 1], p=[0.05, 0.95])

        else:  # Subprime High-Risk
            income[i] = np.random.normal(32_000, 10_000)
            credit_score[i] = np.random.normal(580, 55)
            employment_years[i] = np.random.gamma(2, 1.5)
            debt_to_income[i] = np.random.beta(6, 4) * 0.45 + 0.2
            loan_history_count[i] = np.random.poisson(5)
            age[i] = np.random.normal(30, 7)
            home_ownership[i] = np.random.choice(['rent', 'own'], p=[0.85, 0.15])
            verified_income[i] = np.random.choice([0, 1], p=[0.8, 0.2])

    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_history_count': loan_history_count,
        'age': age,
        'home_ownership': home_ownership,
        'verified_income': verified_income,
    })

    df['income'] = df['income'].clip(lower=0)
    df['credit_score'] = df['credit_score'].clip(lower=300, upper=850)
    df['employment_years'] = df['employment_years'].clip(lower=0)
    df['debt_to_income'] = df['debt_to_income'].clip(lower=0, upper=1)
    df['loan_history_count'] = df['loan_history_count'].astype(int)
    df['age'] = df['age'].clip(lower=18, upper=80)

    return df


if __name__ == '__main__':
    df = generate_customer_data()
    print(df.describe())
    print(df['home_ownership'].value_counts())