"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer data with 4 realistic loan segments.

    Segments:
        0 — Mass Market: moderate income, fair credit, standard DTI
        1 — Rising Prime: growing income, improving credit, low DTI
        2 — Established Prime: high income, excellent credit, very low DTI
        3 — Subprime High-Risk: low income, thin file, high DTI
    """
    segments = []
    for label in range(4):
        size = n // 4
        if label == 0:
            income = np.random.normal(55_000, 15_000, size)
            credit = np.random.normal(660, 60, size)
            employment = np.random.uniform(1, 10, size)
            dti = np.random.uniform(0.2, 0.4, size)
            loans = np.random.poisson(1.5, size)
            age = np.random.randint(22, 50, size)
            home_owner = np.random.choice([0, 1], size, p=[0.65, 0.35])
            verified = np.random.choice([0, 1], size, p=[0.3, 0.7])
        elif label == 1:
            income = np.random.normal(85_000, 20_000, size)
            credit = np.random.normal(730, 50, size)
            employment = np.random.uniform(3, 15, size)
            dti = np.random.uniform(0.15, 0.35, size)
            loans = np.random.poisson(2.0, size)
            age = np.random.randint(26, 55, size)
            home_owner = np.random.choice([0, 1], size, p=[0.45, 0.55])
            verified = np.random.choice([0, 1], size, p=[0.15, 0.85])
        elif label == 2:
            income = np.random.normal(140_000, 35_000, size)
            credit = np.random.normal(790, 40, size)
            employment = np.random.uniform(5, 25, size)
            dti = np.random.uniform(0.08, 0.28, size)
            loans = np.random.poisson(3.0, size)
            age = np.random.randint(32, 65, size)
            home_owner = np.random.choice([0, 1], size, p=[0.20, 0.80])
            verified = np.random.choice([0, 1], size, p=[0.05, 0.95])
        else:
            income = np.random.normal(32_000, 12_000, size)
            credit = np.random.normal(580, 55, size)
            employment = np.random.uniform(0, 5, size)
            dti = np.random.uniform(0.40, 0.65, size)
            loans = np.random.poisson(0.8, size)
            age = np.random.randint(18, 40, size)
            home_owner = np.random.choice([0, 1], size, p=[0.80, 0.20])
            verified = np.random.choice([0, 1], size, p=[0.70, 0.30])

        segments.append(
            pd.DataFrame(
                {
                    "income": income,
                    "credit_score": credit,
                    "employment_years": employment,
                    "debt_to_income": dti,
                    "loan_history_count": loans,
                    "age": age,
                    "home_ownership": home_owner,
                    "verified_income": verified,
                    "segment_label": label,
                }
            )
        )

    df = pd.concat(segments, ignore_index=True)

    # Clip to realistic bounds
    df["income"] = df["income"].clip(lower=10_000)
    df["credit_score"] = df["credit_score"].clip(lower=300, upper=850).round().astype(int)
    df["employment_years"] = df["employment_years"].clip(lower=0)
    df["debt_to_income"] = df["debt_to_income"].clip(lower=0.0, upper=1.0)
    df["loan_history_count"] = df["loan_history_count"].clip(lower=0).astype(int)
    df["age"] = df["age"].clip(lower=18).astype(int)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.describe())
    print(df["segment_label"].value_counts().sort_index())