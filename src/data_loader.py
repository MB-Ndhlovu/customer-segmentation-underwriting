import numpy as np
import pandas as pd

def generate_customer_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # 4 segments with distinct distributions
    # Segment 0: Mass Market  -- young, low credit, moderate DTI
    # Segment 1: Rising Prime -- mid career, growing income, improving credit
    # Segment 2: Established Prime -- high income, excellent credit, long tenure
    # Segment 3: Subprime High-Risk -- low credit, high DTI, thin file

    n_seg = n // 4

    seg_0 = pd.DataFrame({
        "age": rng.integers(18, 30, size=n_seg),
        "income": rng.normal(1_200_000, 200_000, size=n_seg).clip(600_000, 2_000_000),
        "credit_score": rng.integers(520, 640, size=n_seg),
        "employment_years": rng.integers(0, 4, size=n_seg),
        "debt_to_income": rng.uniform(0.15, 0.38, size=n_seg),
        "loan_history_count": rng.integers(0, 3, size=n_seg),
        "home_ownership": rng.choice(["rent", "rent", "parents"], size=n_seg),
        "verified_income": rng.choice([True, False], size=n_seg, p=[0.5, 0.5]),
        "_true_segment": 0,
    })

    seg_1 = pd.DataFrame({
        "age": rng.integers(26, 40, size=n_seg),
        "income": rng.normal(2_800_000, 400_000, size=n_seg).clip(1_800_000, 4_500_000),
        "credit_score": rng.integers(640, 740, size=n_seg),
        "employment_years": rng.integers(3, 10, size=n_seg),
        "debt_to_income": rng.uniform(0.18, 0.35, size=n_seg),
        "loan_history_count": rng.integers(1, 5, size=n_seg),
        "home_ownership": rng.choice(["rent", "own", "rent"], size=n_seg),
        "verified_income": rng.choice([True, False], size=n_seg, p=[0.75, 0.25]),
        "_true_segment": 1,
    })

    seg_2 = pd.DataFrame({
        "age": rng.integers(35, 58, size=n_seg),
        "income": rng.normal(6_500_000, 1_200_000, size=n_seg).clip(4_000_000, 12_000_000),
        "credit_score": rng.integers(720, 850, size=n_seg),
        "employment_years": rng.integers(8, 25, size=n_seg),
        "debt_to_income": rng.uniform(0.08, 0.25, size=n_seg),
        "loan_history_count": rng.integers(3, 12, size=n_seg),
        "home_ownership": rng.choice(["own", "own", "own"], size=n_seg),
        "verified_income": rng.choice([True, False], size=n_seg, p=[0.95, 0.05]),
        "_true_segment": 2,
    })

    seg_3 = pd.DataFrame({
        "age": rng.integers(20, 45, size=n_seg),
        "income": rng.normal(850_000, 250_000, size=n_seg).clip(450_000, 1_600_000),
        "credit_score": rng.integers(480, 600, size=n_seg),
        "employment_years": rng.integers(0, 6, size=n_seg),
        "debt_to_income": rng.uniform(0.35, 0.60, size=n_seg),
        "loan_history_count": rng.integers(0, 2, size=n_seg),
        "home_ownership": rng.choice(["rent", "rent", "parents"], size=n_seg),
        "verified_income": rng.choice([True, False], size=n_seg, p=[0.3, 0.7]),
        "_true_segment": 3,
    })

    df = pd.concat([seg_0, seg_1, seg_2, seg_3], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["income"] = df["income"].round(0).astype(int)
    df["credit_score"] = df["credit_score"].astype(int)
    df["employment_years"] = df["employment_years"].astype(int)
    df["loan_history_count"] = df["loan_history_count"].astype(int)
    df["age"] = df["age"].astype(int)
    df["verified_income"] = df["verified_income"].astype(bool)

    return df


FEATURE_COLS = [
    "income", "credit_score", "employment_years",
    "debt_to_income", "loan_history_count", "age",
    "home_ownership", "verified_income",
]


def load_data():
    return generate_customer_data()