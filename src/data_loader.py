import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)

    # Segment probabilities (balanced-ish with skew toward mass market)
    # 0: Mass Market, 1: Rising Prime, 2: Established Prime, 3: Subprime High-Risk
    segment_probs = [0.35, 0.25, 0.25, 0.15]
    segments = np.random.choice([0, 1, 2, 3], size=n, p=segment_probs)

    income, credit_score, employment_years = [], [], []
    debt_to_income, loan_history_count, age = [], [], []
    home_ownership, verified_income = [], []

    for seg in segments:
        if seg == 0:  # Mass Market
            income.append(np.random.normal(45000, 8000))
            credit_score.append(np.random.normal(660, 50))
            employment_years.append(np.random.exponential(3))
            debt_to_income.append(np.random.uniform(0.15, 0.35))
            loan_history_count.append(np.random.poisson(2))
            age.append(np.random.randint(22, 45))
            home_ownership.append(np.random.choice([0, 1], p=[0.7, 0.3]))
            verified_income.append(np.random.choice([0, 1], p=[0.5, 0.5]))

        elif seg == 1:  # Rising Prime
            income.append(np.random.normal(72000, 12000))
            credit_score.append(np.random.normal(720, 45))
            employment_years.append(np.random.exponential(5))
            debt_to_income.append(np.random.uniform(0.10, 0.28))
            loan_history_count.append(np.random.poisson(3))
            age.append(np.random.randint(25, 50))
            home_ownership.append(np.random.choice([0, 1], p=[0.45, 0.55]))
            verified_income.append(np.random.choice([0, 1], p=[0.3, 0.7]))

        elif seg == 2:  # Established Prime
            income.append(np.random.normal(110000, 20000))
            credit_score.append(np.random.normal(780, 40))
            employment_years.append(np.random.exponential(8))
            debt_to_income.append(np.random.uniform(0.08, 0.22))
            loan_history_count.append(np.random.poisson(4))
            age.append(np.random.randint(30, 60))
            home_ownership.append(np.random.choice([0, 1], p=[0.2, 0.8]))
            verified_income.append(np.random.choice([0, 1], p=[0.15, 0.85]))

        else:  # Subprime High-Risk
            income.append(np.random.normal(32000, 7000))
            credit_score.append(np.random.normal(580, 45))
            employment_years.append(np.random.exponential(2))
            debt_to_income.append(np.random.uniform(0.30, 0.55))
            loan_history_count.append(np.random.poisson(5))
            age.append(np.random.randint(20, 40))
            home_ownership.append(np.random.choice([0, 1], p=[0.85, 0.15]))
            verified_income.append(np.random.choice([0, 1], p=[0.75, 0.25]))

    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_history_count': loan_history_count,
        'age': age,
        'home_ownership': home_ownership,
        'verified_income': verified_income,
        'segment_label': segments
    })

    # Clip extreme values
    df['income'] = df['income'].clip(15000, 300000)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 0.7)
    df['employment_years'] = df['employment_years'].clip(0, 40)
    df['loan_history_count'] = df['loan_history_count'].clip(0, 15)
    df['age'] = df['age'].clip(18, 75)

    return df


def load_data(path=None):
    if path:
        return pd.read_csv(path)
    return generate_synthetic_data()