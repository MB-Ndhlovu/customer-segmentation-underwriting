import numpy as np
import pandas as pd

def generate_customer_data(n=5000, seed=42):
    np.random.seed(seed)
    
    # Segment distributions — we seed specific characteristics per segment
    # to ensure KMeans recovers meaningful groups
    segments = np.random.choice([0, 1, 2, 3], size=n, p=[0.30, 0.25, 0.25, 0.20])
    
    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership = np.zeros(n)  # 0 = rent, 1 = owned
    verified_income = np.zeros(n)  # 0 = unverified, 1 = verified

    for seg in [0, 1, 2, 3]:
        mask = segments == seg
        n_seg = mask.sum()

        if seg == 0:  # Mass Market
            income[mask] = np.random.normal(48000, 12000, n_seg)
            credit_score[mask] = np.random.normal(660, 55, n_seg)
            employment_years[mask] = np.random.exponential(3.5, n_seg)
            debt_to_income[mask] = np.random.normal(0.30, 0.10, n_seg)
            loan_history_count[mask] = np.random.poisson(2.0, n_seg)
            age[mask] = np.random.normal(32, 8, n_seg)
            home_ownership[mask] = np.random.choice([0, 1], n_seg, p=[0.75, 0.25])
            verified_income[mask] = np.random.choice([0, 1], n_seg, p=[0.50, 0.50])

        elif seg == 1:  # Rising Prime
            income[mask] = np.random.normal(78000, 18000, n_seg)
            credit_score[mask] = np.random.normal(720, 45, n_seg)
            employment_years[mask] = np.random.exponential(5.0, n_seg)
            debt_to_income[mask] = np.random.normal(0.24, 0.08, n_seg)
            loan_history_count[mask] = np.random.poisson(2.5, n_seg)
            age[mask] = np.random.normal(36, 7, n_seg)
            home_ownership[mask] = np.random.choice([0, 1], n_seg, p=[0.55, 0.45])
            verified_income[mask] = np.random.choice([0, 1], n_seg, p=[0.35, 0.65])

        elif seg == 2:  # Established Prime
            income[mask] = np.random.normal(115000, 28000, n_seg)
            credit_score[mask] = np.random.normal(780, 40, n_seg)
            employment_years[mask] = np.random.exponential(9.0, n_seg)
            debt_to_income[mask] = np.random.normal(0.18, 0.07, n_seg)
            loan_history_count[mask] = np.random.poisson(3.5, n_seg)
            age[mask] = np.random.normal(44, 9, n_seg)
            home_ownership[mask] = np.random.choice([0, 1], n_seg, p=[0.20, 0.80])
            verified_income[mask] = np.random.choice([0, 1], n_seg, p=[0.15, 0.85])

        elif seg == 3:  # Subprime High-Risk
            income[mask] = np.random.normal(32000, 10000, n_seg)
            credit_score[mask] = np.random.normal(590, 50, n_seg)
            employment_years[mask] = np.random.exponential(1.8, n_seg)
            debt_to_income[mask] = np.random.normal(0.42, 0.12, n_seg)
            loan_history_count[mask] = np.random.poisson(3.5, n_seg)
            age[mask] = np.random.normal(28, 6, n_seg)
            home_ownership[mask] = np.random.choice([0, 1], n_seg, p=[0.88, 0.12])
            verified_income[mask] = np.random.choice([0, 1], n_seg, p=[0.70, 0.30])

    # Clip to realistic bounds
    income = np.clip(income, 12000, 300000)
    credit_score = np.clip(credit_score, 500, 850)
    employment_years = np.clip(employment_years, 0, 40)
    debt_to_income = np.clip(debt_to_income, 0.05, 0.60)
    loan_history_count = np.clip(loan_history_count, 0, 15)
    age = np.clip(age, 18, 75)

    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'loan_history_count': loan_history_count,
        'age': age,
        'home_ownership': home_ownership,
        'verified_income': verified_income,
        'segment_true': segments
    })

    return df

if __name__ == '__main__':
    df = generate_customer_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df['segment_true'].value_counts().sort_index())