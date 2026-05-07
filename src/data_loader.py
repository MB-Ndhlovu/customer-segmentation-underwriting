"""Synthetic customer dataset for underwriting segmentation."""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic customer records with realistic underwriting features.

    Features:
        income: annual income in ZAR
        credit_score: 300–850
        employment_years: 0–40
        debt_to_income: 0.0–1.0
        loan_history_count: 0–15
        age: 18–70
        home_ownership: 0=renting, 1=owning, 2=owned (mortgage paid)
        verified_income: 0=no, 1=yes
        segment_label: 0–3 (assigned by clustering, not here)
    """

    # Four segment pools — we'll sample from each to create natural clusters
    pools = []

    # Segment 0 — Mass Market: low income, fair credit, short employment, high DTI, young
    n0 = int(n * 0.30)
    pools.append({
        'income': np.random.normal(120000, 40000, n0).clip(30000, 250000),
        'credit_score': np.random.normal(580, 60, n0).clip(400, 700),
        'employment_years': np.random.exponential(1.5, n0).clip(0, 8),
        'debt_to_income': np.random.beta(6, 4, n0).clip(0.25, 0.60),
        'loan_history_count': np.random.poisson(1.5, n0).clip(0, 6),
        'age': np.random.normal(26, 5, n0).clip(18, 45),
        'home_ownership': np.random.choice([0, 1, 2], n0, p=[0.75, 0.15, 0.10]),
        'verified_income': np.random.choice([0, 1], n0, p=[0.70, 0.30]),
    })

    # Segment 1 — Rising Prime: moderate income, good credit, stable employment, moderate DTI
    n1 = int(n * 0.30)
    pools.append({
        'income': np.random.normal(280000, 70000, n1).clip(120000, 500000),
        'credit_score': np.random.normal(680, 50, n1).clip(580, 780),
        'employment_years': np.random.exponential(4, n1).clip(1, 15),
        'debt_to_income': np.random.beta(4, 6, n1).clip(0.10, 0.40),
        'loan_history_count': np.random.poisson(2.5, n1).clip(0, 8),
        'age': np.random.normal(34, 7, n1).clip(24, 55),
        'home_ownership': np.random.choice([0, 1, 2], n1, p=[0.40, 0.40, 0.20]),
        'verified_income': np.random.choice([0, 1], n1, p=[0.40, 0.60]),
    })

    # Segment 2 — Established Prime: high income, excellent credit, long tenure, low DTI
    n2 = int(n * 0.25)
    pools.append({
        'income': np.random.normal(550000, 150000, n2).clip(280000, 900000),
        'credit_score': np.random.normal(760, 45, n2).clip(680, 850),
        'employment_years': np.random.exponential(10, n2).clip(5, 35),
        'debt_to_income': np.random.beta(2, 8, n2).clip(0.02, 0.28),
        'loan_history_count': np.random.poisson(4, n2).clip(1, 12),
        'age': np.random.normal(42, 9, n2).clip(30, 65),
        'home_ownership': np.random.choice([0, 1, 2], n2, p=[0.15, 0.35, 0.50]),
        'verified_income': np.random.choice([0, 1], n2, p=[0.15, 0.85]),
    })

    # Segment 3 — Subprime High-Risk: very low income, poor credit, unstable employment, very high DTI
    n3 = n - n0 - n1 - n2
    pools.append({
        'income': np.random.normal(60000, 25000, n3).clip(20000, 120000),
        'credit_score': np.random.normal(480, 55, n3).clip(300, 580),
        'employment_years': np.random.exponential(0.8, n3).clip(0, 4),
        'debt_to_income': np.random.beta(8, 3, n3).clip(0.45, 0.85),
        'loan_history_count': np.random.poisson(4, n3).clip(1, 15),
        'age': np.random.normal(30, 8, n3).clip(18, 55),
        'home_ownership': np.random.choice([0, 1, 2], n3, p=[0.85, 0.10, 0.05]),
        'verified_income': np.random.choice([0, 1], n3, p=[0.90, 0.10]),
    })

    # Concatenate and shuffle
    dfs = [pd.DataFrame(p) for p in pools]
    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Round and type-cast
    df['age'] = df['age'].round().astype(int)
    df['loan_history_count'] = df['loan_history_count'].round().astype(int)
    df['home_ownership'] = df['home_ownership'].astype(int)
    df['verified_income'] = df['verified_income'].astype(int)

    return df


if __name__ == '__main__':
    df = generate_customer_data(5000)
    print(f"Generated {len(df)} records")
    print(df.describe())
    print(df['home_ownership'].value_counts().sort_index())
    print(df['verified_income'].value_counts().sort_index())