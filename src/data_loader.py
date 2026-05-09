import numpy as np
import pandas as pd

def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Segment proportions (roughly): Mass Market (35%), Rising Prime (30%), Established Prime (20%), Subprime High-Risk (15%)
    segment_probs = [0.35, 0.30, 0.20, 0.15]
    segment_labels = np.random.choice(4, size=n, p=segment_probs)

    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership = np.zeros(n)  # 0 = rent, 1 = own
    verified_income = np.zeros(n)  # 0 = unverified, 1 = verified

    # Segment 0: Mass Market
    mask0 = segment_labels == 0
    income[mask0] = np.random.normal(45000, 10000, mask0.sum())
    credit_score[mask0] = np.random.normal(620, 50, mask0.sum())
    employment_years[mask0] = np.random.exponential(4, mask0.sum())
    debt_to_income[mask0] = np.random.normal(0.30, 0.10, mask0.sum())
    loan_history_count[mask0] = np.random.poisson(2, mask0.sum())
    age[mask0] = np.random.normal(32, 8, mask0.sum())
    home_ownership[mask0] = np.random.choice([0, 1], size=mask0.sum(), p=[0.6, 0.4])
    verified_income[mask0] = np.random.choice([0, 1], size=mask0.sum(), p=[0.5, 0.5])

    # Segment 1: Rising Prime
    mask1 = segment_labels == 1
    income[mask1] = np.random.normal(72000, 15000, mask1.sum())
    credit_score[mask1] = np.random.normal(700, 40, mask1.sum())
    employment_years[mask1] = np.random.exponential(6, mask1.sum())
    debt_to_income[mask1] = np.random.normal(0.22, 0.08, mask1.sum())
    loan_history_count[mask1] = np.random.poisson(3, mask1.sum())
    age[mask1] = np.random.normal(36, 7, mask1.sum())
    home_ownership[mask1] = np.random.choice([0, 1], size=mask1.sum(), p=[0.35, 0.65])
    verified_income[mask1] = np.random.choice([0, 1], size=mask1.sum(), p=[0.2, 0.8])

    # Segment 2: Established Prime
    mask2 = segment_labels == 2
    income[mask2] = np.random.normal(110000, 25000, mask2.sum())
    credit_score[mask2] = np.random.normal(770, 35, mask2.sum())
    employment_years[mask2] = np.random.exponential(10, mask2.sum())
    debt_to_income[mask2] = np.random.normal(0.15, 0.06, mask2.sum())
    loan_history_count[mask2] = np.random.poisson(4, mask2.sum())
    age[mask2] = np.random.normal(45, 9, mask2.sum())
    home_ownership[mask2] = np.random.choice([0, 1], size=mask2.sum(), p=[0.15, 0.85])
    verified_income[mask2] = np.random.choice([0, 1], size=mask2.sum(), p=[0.1, 0.9])

    # Segment 3: Subprime High-Risk
    mask3 = segment_labels == 3
    income[mask3] = np.random.normal(28000, 8000, mask3.sum())
    credit_score[mask3] = np.random.normal(560, 45, mask3.sum())
    employment_years[mask3] = np.random.exponential(2.5, mask3.sum())
    debt_to_income[mask3] = np.random.normal(0.45, 0.12, mask3.sum())
    loan_history_count[mask3] = np.random.poisson(4, mask3.sum())
    age[mask3] = np.random.normal(28, 7, mask3.sum())
    home_ownership[mask3] = np.random.choice([0, 1], size=mask3.sum(), p=[0.8, 0.2])
    verified_income[mask3] = np.random.choice([0, 1], size=mask3.sum(), p=[0.7, 0.3])

    df = pd.DataFrame({
        "income": np.clip(income, 5000, 500000),
        "credit_score": np.clip(credit_score, 300, 850),
        "employment_years": np.clip(employment_years, 0, 40),
        "debt_to_income": np.clip(debt_to_income, 0.01, 0.95),
        "loan_history_count": np.clip(loan_history_count, 0, 20).astype(int),
        "age": np.clip(age, 18, 80).astype(int),
        "home_ownership": home_ownership.astype(int),
        "verified_income": verified_income.astype(int),
        "segment_label": segment_labels,
    })

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df["segment_label"].value_counts().sort_index())