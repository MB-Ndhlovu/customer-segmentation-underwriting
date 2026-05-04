"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd


def generate_customer_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data with 4 distinct underwriting segments.

    Segments:
        0 = Mass Market
        1 = Rising Prime
        2 = Established Prime
        3 = Subprime High-Risk
    """
    np.random.seed(seed)

    segment_names = {
        0: "Mass Market",
        1: "Rising Prime",
        2: "Established Prime",
        3: "Subprime High-Risk",
    }

    n_per_segment = n_samples // 4
    records = []

    for seg_id, seg_name in segment_names.items():
        for _ in range(n_per_segment):
            if seg_id == 0:  # Mass Market
                income = np.random.normal(55_000, 12_000)
                credit_score = np.random.normal(660, 50)
                employment_years = np.random.exponential(4)
                debt_to_income = np.random.normal(0.28, 0.08)
                loan_history_count = np.random.poisson(2)
                age = np.random.randint(25, 60)
                home_ownership = "rent" if np.random.random() < 0.55 else "own"
                verified_income = np.random.choice([True, False], p=[0.6, 0.4])

            elif seg_id == 1:  # Rising Prime
                income = np.random.normal(72_000, 15_000)
                credit_score = np.random.normal(720, 45)
                employment_years = np.random.exponential(2.5)
                debt_to_income = np.random.normal(0.22, 0.07)
                loan_history_count = np.random.poisson(1.5)
                age = np.random.randint(22, 38)
                home_ownership = "rent" if np.random.random() < 0.7 else "own"
                verified_income = np.random.choice([True, False], p=[0.75, 0.25])

            elif seg_id == 2:  # Established Prime
                income = np.random.normal(105_000, 25_000)
                credit_score = np.random.normal(780, 35)
                employment_years = np.random.exponential(10)
                debt_to_income = np.random.normal(0.18, 0.06)
                loan_history_count = np.random.poisson(3)
                age = np.random.randint(35, 65)
                home_ownership = "own" if np.random.random() < 0.75 else "rent"
                verified_income = np.random.choice([True, False], p=[0.9, 0.1])

            else:  # Subprime High-Risk
                income = np.random.normal(38_000, 10_000)
                credit_score = np.random.normal(590, 55)
                employment_years = np.random.exponential(3)
                debt_to_income = np.random.normal(0.45, 0.12)
                loan_history_count = np.random.poisson(5)
                age = np.random.randint(20, 55)
                home_ownership = "rent" if np.random.random() < 0.8 else "own"
                verified_income = np.random.choice([True, False], p=[0.35, 0.65])

            records.append({
                "income": max(0, income),
                "credit_score": min(max(int(credit_score), 300), 850),
                "employment_years": max(0, employment_years),
                "debt_to_income": max(0, debt_to_income),
                "loan_history_count": max(0, int(loan_history_count)),
                "age": max(18, min(age, 80)),
                "home_ownership": home_ownership,
                "verified_income": verified_income,
                "segment_label": seg_id,
                "segment_name": seg_name,
            })

    df = pd.DataFrame(records)
    df["home_ownership"] = df["home_ownership"].map({"own": 1, "rent": 0})
    df["verified_income"] = df["verified_income"].astype(int)

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(f"Generated {len(df)} rows")
    print(df.groupby("segment_name")[".credit_score", "income", "debt_to_income"].mean())