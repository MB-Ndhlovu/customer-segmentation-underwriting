"""Synthetic customer dataset generator for loan underwriting segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer dataset with 4 distinct segment clusters.

    Segments are seeded via latent variables so KMeans discovers them naturally.
    """
    # Latent segment probabilities (per segment weights)
    segment_weights = [0.35, 0.30, 0.20, 0.15]  # Mass Market, Rising Prime, Established Prime, Subprime HR

    segment_assignments = np.random.choice(4, size=n, p=segment_weights)

    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership = np.zeros(n, dtype=int)
    verified_income = np.zeros(n, dtype=int)

    for seg in range(4):
        mask = segment_assignments == seg
        m = mask.sum()

        if seg == 0:  # Mass Market
            income[mask] = np.random.normal(42000, 12000, m)
            credit_score[mask] = np.random.normal(650, 60, m)
            employment_years[mask] = np.random.exponential(3.5, m)
            debt_to_income[mask] = np.random.normal(0.28, 0.10, m)
            loan_history_count[mask] = np.random.poisson(1.5, m)
            age[mask] = np.random.normal(32, 8, m)
            home_ownership[mask] = np.random.binomial(1, 0.20, m)
            verified_income[mask] = np.random.binomial(1, 0.45, m)

        elif seg == 1:  # Rising Prime
            income[mask] = np.random.normal(68000, 15000, m)
            credit_score[mask] = np.random.normal(715, 55, m)
            employment_years[mask] = np.random.exponential(5.0, m)
            debt_to_income[mask] = np.random.normal(0.22, 0.08, m)
            loan_history_count[mask] = np.random.poisson(2.5, m)
            age[mask] = np.random.normal(38, 7, m)
            home_ownership[mask] = np.random.binomial(1, 0.45, m)
            verified_income[mask] = np.random.binomial(1, 0.72, m)

        elif seg == 2:  # Established Prime
            income[mask] = np.random.normal(110000, 25000, m)
            credit_score[mask] = np.random.normal(780, 45, m)
            employment_years[mask] = np.random.exponential(10.0, m)
            debt_to_income[mask] = np.random.normal(0.15, 0.06, m)
            loan_history_count[mask] = np.random.poisson(3.5, m)
            age[mask] = np.random.normal(45, 8, m)
            home_ownership[mask] = np.random.binomial(1, 0.80, m)
            verified_income[mask] = np.random.binomial(1, 0.93, m)

        elif seg == 3:  # Subprime High-Risk
            income[mask] = np.random.normal(31000, 10000, m)
            credit_score[mask] = np.random.normal(570, 70, m)
            employment_years[mask] = np.random.exponential(1.8, m)
            debt_to_income[mask] = np.random.normal(0.45, 0.14, m)
            loan_history_count[mask] = np.random.poisson(4.0, m)
            age[mask] = np.random.normal(28, 6, m)
            home_ownership[mask] = np.random.binomial(1, 0.08, m)
            verified_income[mask] = np.random.binomial(1, 0.22, m)

    # Clamp to realistic bounds
    income = np.clip(income, 15000, 500000)
    credit_score = np.clip(credit_score, 300, 850)
    employment_years = np.clip(employment_years, 0, 45)
    debt_to_income = np.clip(debt_to_income, 0.01, 0.80)
    loan_history_count = np.clip(loan_history_count, 0, 20)
    age = np.clip(age, 18, 75)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership": home_ownership,
        "verified_income": verified_income,
        "_segment_true": segment_assignments,
    })

    # Round
    df["income"] = df["income"].round(2)
    df["credit_score"] = df["credit_score"].round(1)
    df["employment_years"] = df["employment_years"].round(2)
    df["debt_to_income"] = df["debt_to_income"].round(4)
    df["loan_history_count"] = df["loan_history_count"].astype(int)
    df["age"] = df["age"].astype(int)

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df["_segment_true"].value_counts().sort_index())