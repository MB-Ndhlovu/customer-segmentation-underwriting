"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

SEGMENT_PARAMS = {
    # income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership (owner prob), verified_income (prob)
    0: dict(income_mean=65000, income_std=15000, credit_mean=680, credit_std=60,
            emp_mean=5, emp_std=2, dti_mean=0.28, dti_std=0.08, loan_mean=2, loan_std=1,
            age_mean=35, age_std=8, home_prob=0.45, verified_prob=0.55),
    1: dict(income_mean=82000, income_std=20000, credit_mean=720, credit_std=50,
            emp_mean=3, emp_std=1.5, dti_mean=0.22, dti_std=0.07, loan_mean=1, loan_std=1,
            age_mean=28, age_std=4, home_prob=0.25, verified_prob=0.65),
    2: dict(income_mean=140000, income_std=40000, credit_mean=780, credit_std=40,
            emp_mean=12, emp_std=5, dti_mean=0.18, dti_std=0.06, loan_mean=3, loan_std=2,
            age_mean=45, age_std=8, home_prob=0.80, verified_prob=0.90),
    3: dict(income_mean=38000, income_std=10000, credit_mean=580, credit_std=55,
            emp_mean=1.5, emp_std=1, dti_mean=0.42, dti_std=0.10, loan_mean=4, loan_std=2,
            age_mean=26, age_std=6, home_prob=0.10, verified_prob=0.20),
}


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer records across 4 segments."""
    rows = []
    per_segment = n // 4

    for seg_id, params in SEGMENT_PARAMS.items():
        for _ in range(per_segment):
            age = max(18, np.random.normal(params["age_mean"], params["age_std"]))
            emp = max(0, np.random.normal(params["emp_mean"], params["emp_std"]))
            income = max(15000, np.random.normal(params["income_mean"], params["income_std"]))
            credit = int(np.clip(np.random.normal(params["credit_mean"], params["credit_std"]), 300, 850))
            dti = np.clip(np.random.normal(params["dti_mean"], params["dti_std"]), 0.05, 0.95)
            loans = max(0, int(np.random.normal(params["loan_mean"], params["loan_std"])))
            home = 1 if np.random.random() < params["home_prob"] else 0
            verified = 1 if np.random.random() < params["verified_prob"] else 0

            rows.append({
                "income": round(income, 2),
                "credit_score": credit,
                "employment_years": round(emp, 2),
                "debt_to_income": round(dti, 4),
                "loan_history_count": loans,
                "age": int(age),
                "home_ownership": home,
                "verified_income": verified,
                "_segment_hint": seg_id,
            })

    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Drop helper column
    df.drop(columns=["_segment_hint"], inplace=True)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix for clustering / classification."""
    return df.drop(columns=["verified_income"])


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.describe())
    print(df.head())