import numpy as np
import pandas as pd

np.random.seed(42)

def generate_customer_data(n=5000):
    """Generate synthetic customer records with 4 distinct underwriting segments."""
    segments = {
        0: dict(income=(45000, 80000), credit_score=(620, 720),
                employment_years=(2, 10), debt_to_income=(0.15, 0.35),
                loan_history_count=(1, 5), age=(25, 55),
                home_ownership_prob=0.30, verified_income_prob=0.40),
        1: dict(income=(80000, 140000), credit_score=(700, 800),
                employment_years=(5, 20), debt_to_income=(0.10, 0.30),
                loan_history_count=(2, 8), age=(30, 50),
                home_ownership_prob=0.60, verified_income_prob=0.75),
        2: dict(income=(140000, 300000), credit_score=(780, 850),
                employment_years=(10, 35), debt_to_income=(0.05, 0.22),
                loan_history_count=(3, 12), age=(35, 60),
                home_ownership_prob=0.90, verified_income_prob=0.98),
        3: dict(income=(20000, 50000), credit_score=(450, 620),
                employment_years=(0, 3), debt_to_income=(0.35, 0.65),
                loan_history_count=(0, 3), age=(20, 40),
                home_ownership_prob=0.10, verified_income_prob=0.15),
    }

    rows = []
    counts = [1250, 1250, 1250, 1250]
    for seg_id, params in segments.items():
        n_seg = counts[seg_id]
        for _ in range(n_seg):
            income = np.random.uniform(*params['income'])
            credit_score = int(np.random.uniform(*params['credit_score']))
            employment_years = np.random.uniform(*params['employment_years'])
            dti = np.random.uniform(*params['debt_to_income'])
            loan_count = int(np.random.uniform(*params['loan_history_count']))
            age = int(np.random.uniform(*params['age']))
            home_ownership = 1 if np.random.random() < params['home_ownership_prob'] else 0
            verified_income = 1 if np.random.random() < params['verified_income_prob'] else 0

            rows.append({
                'income': round(income, 2),
                'credit_score': credit_score,
                'employment_years': round(employment_years, 2),
                'debt_to_income': round(dti, 4),
                'loan_history_count': loan_count,
                'age': age,
                'home_ownership': home_ownership,
                'verified_income': verified_income,
                'true_segment': seg_id,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_customer_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nTrue segment distribution:\n{df['true_segment'].value_counts().sort_index()}")