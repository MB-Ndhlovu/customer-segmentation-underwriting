import numpy as np
import pandas as pd

np.random.seed(42)

def generate_customer_data(n=5000):
    # Assign segment labels first to ensure distinct clusters
    # 0: Mass Market, 1: Rising Prime, 2: Established Prime, 3: Subprime High-Risk
    segment_probs = [0.35, 0.30, 0.20, 0.15]
    segment_labels = np.random.choice(4, size=n, p=segment_probs)

    income, credit_score, employment_years = [], [], []
    debt_to_income, loan_history_count, age = [], [], []
    home_ownership, verified_income = [], []

    for seg in segment_labels:
        if seg == 0:  # Mass Market
            income.append(np.random.normal(45000, 8000))
            credit_score.append(np.random.normal(660, 50))
            employment_years.append(np.random.exponential(2.5))
            debt_to_income.append(np.random.normal(0.32, 0.08))
            loan_history_count.append(np.random.poisson(1.5))
            age.append(np.random.randint(22, 40))
            home_ownership.append(np.random.choice([0, 1], p=[0.75, 0.25]))
            verified_income.append(np.random.choice([0, 1], p=[0.4, 0.6]))

        elif seg == 1:  # Rising Prime
            income.append(np.random.normal(72000, 10000))
            credit_score.append(np.random.normal(720, 40))
            employment_years.append(np.random.exponential(4.0))
            debt_to_income.append(np.random.normal(0.25, 0.07))
            loan_history_count.append(np.random.poisson(2.5))
            age.append(np.random.randint(28, 45))
            home_ownership.append(np.random.choice([0, 1], p=[0.45, 0.55]))
            verified_income.append(np.random.choice([0, 1], p=[0.2, 0.8]))

        elif seg == 2:  # Established Prime
            income.append(np.random.normal(120000, 20000))
            credit_score.append(np.random.normal(780, 35))
            employment_years.append(np.random.exponential(8.0))
            debt_to_income.append(np.random.normal(0.18, 0.06))
            loan_history_count.append(np.random.poisson(3.5))
            age.append(np.random.randint(35, 60))
            home_ownership.append(np.random.choice([0, 1], p=[0.15, 0.85]))
            verified_income.append(np.random.choice([0, 1], p=[0.05, 0.95]))

        else:  # Subprime High-Risk
            income.append(np.random.normal(32000, 6000))
            credit_score.append(np.random.normal(580, 45))
            employment_years.append(np.random.exponential(1.5))
            debt_to_income.append(np.random.normal(0.45, 0.10))
            loan_history_count.append(np.random.poisson(0.8))
            age.append(np.random.randint(20, 35))
            home_ownership.append(np.random.choice([0, 1], p=[0.85, 0.15]))
            verified_income.append(np.random.choice([0, 1], p=[0.7, 0.3]))

    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_history_count': loan_history_count,
        'age': age,
        'home_ownership': home_ownership,
        'verified_income': verified_income,
        'segment_label': segment_labels
    })

    # Clip to realistic bounds
    df['income'] = df['income'].clip(15000, 300000)
    df['credit_score'] = df['credit_score'].clip(500, 850)
    df['employment_years'] = df['employment_years'].clip(0, 40)
    df['debt_to_income'] = df['debt_to_income'].clip(0.05, 0.65)
    df['loan_history_count'] = df['loan_history_count'].clip(0, 12)
    df['age'] = df['age'].clip(18, 75)

    return df

if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print(f"\nSegment distribution:\n{df['segment_label'].value_counts().sort_index()}")