"""Synthetic customer dataset generator for lending segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

SEGMENT_PROFILES = {
    0: {  # Mass Market
        "income": (28000, 65000),
        "credit_score": (580, 700),
        "employment_years": (1, 8),
        "debt_to_income": (0.15, 0.35),
        "loan_history_count": (0, 3),
        "age": (21, 35),
        "home_ownership": ["rent", "rent", "rent", "parents"],
        "verified_income": [0, 0, 1],
    },
    1: {  # Rising Prime
        "income": (55000, 110000),
        "credit_score": (680, 760),
        "employment_years": (4, 15),
        "debt_to_income": (0.20, 0.40),
        "loan_history_count": (1, 5),
        "age": (28, 45),
        "home_ownership": ["rent", "rent", "own", "own"],
        "verified_income": [0, 1, 1, 1],
    },
    2: {  # Established Prime
        "income": (90000, 220000),
        "credit_score": (750, 850),
        "employment_years": (10, 35),
        "debt_to_income": (0.10, 0.30),
        "loan_history_count": (2, 6),
        "age": (38, 62),
        "home_ownership": ["own", "own", "own", "mortgage"],
        "verified_income": [1, 1, 1, 1],
    },
    3: {  # Subprime High-Risk
        "income": (18000, 45000),
        "credit_score": (480, 620),
        "employment_years": (0, 5),
        "debt_to_income": (0.35, 0.60),
        "loan_history_count": (3, 9),
        "age": (20, 40),
        "home_ownership": ["rent", "rent", "rent", "parents"],
        "verified_income": [0, 0, 0, 1],
    },
}


def _sample(params, n):
    low, high = params
    return np.random.uniform(low, high, n)


def generate_customer_data(n=5000):
    segment_counts = {
        0: int(n * 0.35),  # Mass Market ~35%
        1: int(n * 0.28),  # Rising Prime ~28%
        2: int(n * 0.22),  # Established Prime ~22%
        3: int(n * 0.15),  # Subprime High-Risk ~15%
    }
    # Adjust to hit exactly n
    diff = n - sum(segment_counts.values())
    segment_counts[0] += diff

    records = []
    for seg_id, count in segment_counts.items():
        p = SEGMENT_PROFILES[seg_id]
        records.append(
            {
                "income": _sample(p["income"], count),
                "credit_score": _sample(p["credit_score"], count),
                "employment_years": _sample(p["employment_years"], count),
                "debt_to_income": _sample(p["debt_to_income"], count),
                "loan_history_count": np.random.randint(p["loan_history_count"][0], p["loan_history_count"][1] + 1, count),
                "age": np.random.randint(p["age"][0], p["age"][1] + 1, count),
                "home_ownership": np.random.choice(p["home_ownership"], count),
                "verified_income": np.random.choice(p["verified_income"], count),
                "segment_true": [seg_id] * count,
            }
        )

    df = pd.DataFrame(
        {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    )
    df["home_ownership"] = LabelEncoder().fit_transform(df["home_ownership"])
    # verified_income already 0/1
    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.shape)
    print(df["segment_true"].value_counts().sort_index())