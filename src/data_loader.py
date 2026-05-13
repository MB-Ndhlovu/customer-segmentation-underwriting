import numpy as np
import pandas as pd

def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)
    data = {}

    # Segment probabilities for realistic distribution
    # 0: Mass Market (45%), 1: Rising Prime (30%), 2: Established Prime (15%), 3: Subprime High-Risk (10%)
    segment_probs = [0.45, 0.30, 0.15, 0.10]
    segments = np.random.choice(4, size=n, p=segment_probs)

    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership = np.zeros(n, dtype=int)
    verified_income = np.zeros(n, dtype=int)

    for seg in range(4):
        mask = segments == seg
        count = mask.sum()

        if seg == 0:  # Mass Market
            income[mask] = np.random.normal(45000, 12000, count)
            credit_score[mask] = np.random.normal(680, 50, count)
            employment_years[mask] = np.random.exponential(3, count)
            debt_to_income[mask] = np.random.normal(0.28, 0.08, count)
            loan_history_count[mask] = np.random.poisson(1.5, count)
            age[mask] = np.random.normal(32, 8, count)
            home_ownership[mask] = np.random.choice([0, 1], size=count, p=[0.75, 0.25])
            verified_income[mask] = np.random.choice([0, 1], size=count, p=[0.60, 0.40])

        elif seg == 1:  # Rising Prime
            income[mask] = np.random.normal(75000, 18000, count)
            credit_score[mask] = np.random.normal(730, 45, count)
            employment_years[mask] = np.random.exponential(5, count)
            debt_to_income[mask] = np.random.normal(0.22, 0.07, count)
            loan_history_count[mask] = np.random.poisson(2.5, count)
            age[mask] = np.random.normal(36, 7, count)
            home_ownership[mask] = np.random.choice([0, 1], size=count, p=[0.55, 0.45])
            verified_income[mask] = np.random.choice([0, 1], size=count, p=[0.30, 0.70])

        elif seg == 2:  # Established Prime
            income[mask] = np.random.normal(120000, 30000, count)
            credit_score[mask] = np.random.normal(780, 35, count)
            employment_years[mask] = np.random.exponential(10, count)
            debt_to_income[mask] = np.random.normal(0.15, 0.05, count)
            loan_history_count[mask] = np.random.poisson(4, count)
            age[mask] = np.random.normal(45, 8, count)
            home_ownership[mask] = np.random.choice([0, 1], size=count, p=[0.20, 0.80])
            verified_income[mask] = np.random.choice([0, 1], size=count, p=[0.10, 0.90])

        elif seg == 3:  # Subprime High-Risk
            income[mask] = np.random.normal(28000, 8000, count)
            credit_score[mask] = np.random.normal(580, 40, count)
            employment_years[mask] = np.random.exponential(2, count)
            debt_to_income[mask] = np.random.normal(0.42, 0.10, count)
            loan_history_count[mask] = np.random.poisson(3, count)
            age[mask] = np.random.normal(28, 6, count)
            home_ownership[mask] = np.random.choice([0, 1], size=count, p=[0.85, 0.15])
            verified_income[mask] = np.random.choice([0, 1], size=count, p=[0.80, 0.20])

    data['income'] = np.clip(income, 15000, 500000)
    data['credit_score'] = np.clip(credit_score, 400, 850)
    data['employment_years'] = np.clip(employment_years, 0, 40)
    data['debt_to_income'] = np.clip(debt_to_income, 0.01, 0.80)
    data['loan_history_count'] = np.clip(loan_history_count, 0, 15)
    data['age'] = np.clip(age, 18, 70)
    data['home_ownership'] = home_ownership
    data['verified_income'] = verified_income
    data['segment_label'] = segments

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Generated {len(df)} rows")
    print(df['segment_label'].value_counts().sort_index())
    print(df.describe())