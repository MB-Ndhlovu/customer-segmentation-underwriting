import numpy as np
import pandas as pd

def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)

    # Segment proportions and characteristics
    # 0: Mass Market (40%), 1: Rising Prime (30%), 2: Established Prime (20%), 3: Subprime High-Risk (10%)
    segment_probs = [0.40, 0.30, 0.20, 0.10]
    segments = np.random.choice(4, size=n, p=segment_probs)

    income, credit_score, employment_years = [], [], []
    debt_to_income, loan_history_count, age = [], [], []
    home_ownership, verified_income = [], []

    for seg in segments:
        if seg == 0:  # Mass Market
            income.append(np.random.lognormal(10.5, 0.4))
            credit_score.append(np.random.randint(580, 680))
            employment_years.append(np.random.exponential(2.5))
            debt_to_income.append(np.random.beta(8, 2) * 0.5 + 0.3)  # 0.30-0.80
            loan_history_count.append(np.random.poisson(2))
            age.append(np.random.randint(22, 35))
            home_ownership.append(0)  # renting
            verified_income.append(np.random.choice([0, 1], p=[0.7, 0.3]))
        elif seg == 1:  # Rising Prime
            income.append(np.random.lognormal(11.2, 0.35))
            credit_score.append(np.random.randint(680, 750))
            employment_years.append(np.random.exponential(4.0))
            debt_to_income.append(np.random.beta(6, 4) * 0.4 + 0.15)  # 0.15-0.55
            loan_history_count.append(np.random.poisson(3))
            age.append(np.random.randint(28, 42))
            home_ownership.append(np.random.choice([0, 1], p=[0.6, 0.4]))
            verified_income.append(np.random.choice([0, 1], p=[0.4, 0.6]))
        elif seg == 2:  # Established Prime
            income.append(np.random.lognormal(12.0, 0.3))
            credit_score.append(np.random.randint(750, 850))
            employment_years.append(np.random.exponential(8.0))
            debt_to_income.append(np.random.beta(3, 7) * 0.3 + 0.05)  # 0.05-0.35
            loan_history_count.append(np.random.poisson(4))
            age.append(np.random.randint(35, 60))
            home_ownership.append(1)  # homeowner
            verified_income.append(np.random.choice([0, 1], p=[0.1, 0.9]))
        else:  # Subprime High-Risk
            income.append(np.random.lognormal(10.0, 0.5))
            credit_score.append(np.random.randint(300, 620))
            employment_years.append(np.random.exponential(1.5))
            debt_to_income.append(np.random.beta(9, 1) * 0.4 + 0.5)  # 0.50-0.90
            loan_history_count.append(np.random.poisson(5))
            age.append(np.random.randint(20, 50))
            home_ownership.append(np.random.choice([0, 1], p=[0.85, 0.15]))
            verified_income.append(np.random.choice([0, 1], p=[0.8, 0.2]))

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

    # Clip outliers
    df['income'] = df['income'].clip(upper=500000)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['debt_to_income'] = df['debt_to_income'].clip(0.01, 0.99)
    df['employment_years'] = df['employment_years'].clip(0, 50)
    df['loan_history_count'] = df['loan_history_count'].clip(0, 30)

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    print(df['segment_label'].value_counts().sort_index())