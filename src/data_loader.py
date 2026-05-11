"""Synthetic customer dataset for underwriting segmentation."""
import numpy as np
import pandas as pd

np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic customer records."""
    segments = []
    for _ in range(n):
        r = np.random.rand()
        if r < 0.35:
            segments.append(0)   # Mass Market
        elif r < 0.60:
            segments.append(1)   # Rising Prime
        elif r < 0.80:
            segments.append(2)   # Established Prime
        else:
            segments.append(3)   # Subprime High-Risk

    records = []
    for seg in segments:
        if seg == 0:  # Mass Market
            income = np.random.normal(52000, 12000)
            credit_score = np.random.normal(645, 55)
            employment_years = np.random.normal(5, 3)
            debt_to_income = np.random.normal(0.28, 0.08)
            loan_history_count = np.random.randint(0, 4)
            age = np.random.randint(22, 55)
            home_ownership = np.random.choice(["rent", "own", "rent"], p=[0.6, 0.25, 0.15])
            verified_income = np.random.choice([True, False], p=[0.55, 0.45])
        elif seg == 1:  # Rising Prime
            income = np.random.normal(78000, 18000)
            credit_score = np.random.normal(705, 45)
            employment_years = np.random.normal(3.5, 2)
            debt_to_income = np.random.normal(0.22, 0.07)
            loan_history_count = np.random.randint(1, 5)
            age = np.random.randint(24, 40)
            home_ownership = np.random.choice(["rent", "own", "rent"], p=[0.55, 0.30, 0.15])
            verified_income = np.random.choice([True, False], p=[0.65, 0.35])
        elif seg == 2:  # Established Prime
            income = np.random.normal(135000, 30000)
            credit_score = np.random.normal(765, 40)
            employment_years = np.random.normal(12, 5)
            debt_to_income = np.random.normal(0.18, 0.06)
            loan_history_count = np.random.randint(2, 8)
            age = np.random.randint(32, 60)
            home_ownership = np.random.choice(["own", "own", "rent"], p=[0.70, 0.20, 0.10])
            verified_income = np.random.choice([True, False], p=[0.85, 0.15])
        else:  # Subprime High-Risk
            income = np.random.normal(32000, 9000)
            credit_score = np.random.normal(565, 50)
            employment_years = np.random.normal(3, 2.5)
            debt_to_income = np.random.normal(0.42, 0.10)
            loan_history_count = np.random.randint(0, 7)
            age = np.random.randint(20, 50)
            home_ownership = np.random.choice(["rent", "rent", "own"], p=[0.75, 0.15, 0.10])
            verified_income = np.random.choice([True, False], p=[0.30, 0.70])

        records.append({
            "income": max(income, 5000),
            "credit_score": min(max(credit_score, 300), 850),
            "employment_years": max(employment_years, 0),
            "debt_to_income": max(debt_to_income, 0.01),
            "loan_history_count": loan_history_count,
            "age": age,
            "home_ownership": home_ownership,
            "verified_income": verified_income,
            "segment_true": seg,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_customer_data(5000)
    print(df.head())
    print(df["segment_true"].value_counts().sort_index())