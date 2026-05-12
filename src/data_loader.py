"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_PARAMS = {
    0: {  # Mass Market
        "income": (35000, 65000),
        "credit_score": (580, 680),
        "employment_years": (1, 8),
        "debt_to_income": (0.15, 0.35),
        "loan_history_count": (0, 4),
        "age": (22, 45),
        "home_ownership_rent": 0.70,
        "verified_income_prob": 0.55,
    },
    1: {  # Rising Prime
        "income": (55000, 95000),
        "credit_score": (660, 740),
        "employment_years": (3, 10),
        "debt_to_income": (0.10, 0.28),
        "loan_history_count": (1, 5),
        "age": (26, 42),
        "home_ownership_rent": 0.45,
        "verified_income_prob": 0.72,
    },
    2: {  # Established Prime
        "income": (85000, 180000),
        "credit_score": (720, 820),
        "employment_years": (7, 25),
        "debt_to_income": (0.05, 0.22),
        "loan_history_count": (2, 7),
        "age": (32, 60),
        "home_ownership_rent": 0.20,
        "verified_income_prob": 0.90,
    },
    3: {  # Subprime High-Risk
        "income": (20000, 48000),
        "credit_score": (480, 600),
        "employment_years": (0, 4),
        "debt_to_income": (0.30, 0.55),
        "loan_history_count": (2, 8),
        "age": (19, 38),
        "home_ownership_rent": 0.85,
        "verified_income_prob": 0.30,
    },
}

SEGMENT_WEIGHTS = [0.35, 0.28, 0.22, 0.15]  # realistic market distribution


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer records for 4 underwriting segments."""
    rows_per_segment = np.random.multinomial(n, SEGMENT_WEIGHTS)

    records = []
    for seg_id, count in enumerate(rows_per_segment):
        p = SEGMENT_PARAMS[seg_id]
        for _ in range(count):
            income = np.random.uniform(*p["income"])
            credit_score = int(np.random.uniform(*p["credit_score"]))
            employment_years = np.random.uniform(*p["employment_years"])
            debt_to_income = np.random.uniform(*p["debt_to_income"])
            loan_history_count = int(np.random.uniform(*p["loan_history_count"]))
            age = int(np.random.uniform(*p["age"]))
            home_ownership = 0 if np.random.random() < p["home_ownership_rent"] else 1
            verified_income = 1 if np.random.random() < p["verified_income_prob"] else 0

            records.append(
                {
                    "income": round(income, 2),
                    "credit_score": credit_score,
                    "employment_years": round(employment_years, 2),
                    "debt_to_income": round(debt_to_income, 4),
                    "loan_history_count": loan_history_count,
                    "age": age,
                    "home_ownership": home_ownership,
                    "verified_income": verified_income,
                }
            )

    df = pd.DataFrame(records)
    # slight noise to prevent perfect separability
    df["income"] += np.random.normal(0, df["income"].mean() * 0.02, n)
    df["income"] = df["income"].clip(lower=5000)
    df["debt_to_income"] = df["debt_to_income"].clip(0, 0.95)
    return df


if __name__ == "__main__":
    df = generate_customer_data(5000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSegment counts (approx): {dict(zip(range(4), np.random.multinomial(5000, SEGMENT_WEIGHTS)))}")