"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd

SEGMENT_SEEDS = {
    0: dict(income=(35000, 70000), credit_score=(580, 669), employment_years=(0, 5),
            debt_to_income=(0.15, 0.35), loan_history_count=(0, 2), age=(21, 35),
            home_ownership=0.20, verified_income=0.25),
    1: dict(income=(60000, 100000), credit_score=(670, 719), employment_years=(3, 8),
            debt_to_income=(0.10, 0.28), loan_history_count=(1, 4), age=(25, 40),
            home_ownership=0.45, verified_income=0.55),
    2: dict(income=(90000, 180000), credit_score=(720, 850), employment_years=(5, 20),
            debt_to_income=(0.05, 0.22), loan_history_count=(2, 6), age=(32, 58),
            home_ownership=0.80, verified_income=0.90),
    3: dict(income=(20000, 45000), credit_score=(450, 579), employment_years=(0, 3),
            debt_to_income=(0.30, 0.55), loan_history_count=(3, 10), age=(19, 32),
            home_ownership=0.08, verified_income=0.10),
}

SEGMENT_WEIGHTS = [0.35, 0.25, 0.20, 0.20]

np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic customer records across 4 underwriting segments."""
    rows = []
    for _ in range(n):
        seg = np.random.choice([0, 1, 2, 3], p=SEGMENT_WEIGHTS)
        p = SEGMENT_SEEDS[seg]
        income = np.random.uniform(*p["income"])
        credit_score = int(np.random.uniform(*p["credit_score"]))
        employment_years = np.random.uniform(*p["employment_years"])
        debt_to_income = np.random.uniform(*p["debt_to_income"])
        loan_history_count = int(np.random.uniform(*p["loan_history_count"]))
        age = int(np.random.uniform(*p["age"]))
        home_ownership = 1 if np.random.random() < p["home_ownership"] else 0
        verified_income = 1 if np.random.random() < p["verified_income"] else 0
        rows.append({
            "income": round(income, 2),
            "credit_score": credit_score,
            "employment_years": round(employment_years, 2),
            "debt_to_income": round(debt_to_income, 4),
            "loan_history_count": loan_history_count,
            "age": age,
            "home_ownership": home_ownership,
            "verified_income": verified_income,
            "_true_segment": seg,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_customer_data(5000)
    df.to_csv("customers.csv", index=False)
    print(f"Generated {len(df)} rows → customers.csv")