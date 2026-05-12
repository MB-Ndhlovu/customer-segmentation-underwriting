"""Synthetic customer dataset for loan underwriting segmentation."""

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker(seed=42)
np.random.seed(42)

SEGMENT_PARAMS = {
    0: {  # Mass Market
        "income_range": (25000, 65000),
        "credit_score_range": (620, 720),
        "employment_years_range": (0.5, 5),
        "debt_to_income_range": (0.15, 0.38),
        "loan_history_count_range": (0, 3),
        "age_range": (22, 40),
        "home_ownership_probs": (0.15, 0.30, 0.55),  # rent, own, none
        "verified_income_prob": 0.55,
    },
    1: {  # Rising Prime
        "income_range": (55000, 110000),
        "credit_score_range": (680, 760),
        "employment_years_range": (2, 10),
        "debt_to_income_range": (0.18, 0.35),
        "loan_history_count_range": (1, 5),
        "age_range": (28, 45),
        "home_ownership_probs": (0.35, 0.45, 0.20),
        "verified_income_prob": 0.80,
    },
    2: {  # Established Prime
        "income_range": (90000, 200000),
        "credit_score_range": (740, 850),
        "employment_years_range": (5, 25),
        "debt_to_income_range": (0.10, 0.28),
        "loan_history_count_range": (2, 8),
        "age_range": (35, 62),
        "home_ownership_probs": (0.10, 0.80, 0.10),
        "verified_income_prob": 0.95,
    },
    3: {  # Subprime High-Risk
        "income_range": (18000, 48000),
        "credit_score_range": (500, 640),
        "employment_years_range": (0, 3),
        "debt_to_income_range": (0.35, 0.60),
        "loan_history_count_range": (3, 10),
        "age_range": (20, 45),
        "home_ownership_probs": (0.45, 0.10, 0.45),
        "verified_income_prob": 0.30,
    },
}

SEGMENT_WEIGHTS = [0.38, 0.28, 0.18, 0.16]  # roughly real-world distribution


def _draw_segment_labels(n: int) -> np.ndarray:
    """Draw segment labels respecting realistic distribution."""
    return np.random.choice([0, 1, 2, 3], size=n, p=SEGMENT_WEIGHTS)


def _home_ownership_label(probs: tuple) -> str:
    r = np.random.rand()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return ["rent", "own", "none"][i]
    return "rent"


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic customer records with realistic underwriting features."""
    segment_labels = _draw_segment_labels(n)

    records = []
    for seg in segment_labels:
        p = SEGMENT_PARAMS[seg]
        income = np.random.uniform(*p["income_range"])
        credit_score = int(np.random.uniform(*p["credit_score_range"]))
        employment_years = round(np.random.uniform(*p["employment_years_range"]), 1)
        debt_to_income = round(np.random.uniform(*p["debt_to_income_range"]), 4)
        loan_history_count = int(np.random.uniform(*p["loan_history_count_range"]))
        age = int(np.random.uniform(*p["age_range"]))
        home_ownership = _home_ownership_label(p["home_ownership_probs"])
        verified_income = 1 if np.random.rand() < p["verified_income_prob"] else 0

        records.append({
            "income": round(income, 2),
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "loan_history_count": loan_history_count,
            "age": age,
            "home_ownership": home_ownership,
            "verified_income": verified_income,
            "segment_label": int(seg),
        })

    df = pd.DataFrame(records)
    # Add a little noise to make clusters non-trivial
    df["income"] += np.random.normal(0, df["income"] * 0.02, n)
    df["debt_to_income"] = df["debt_to_income"].clip(0.01, 0.95)
    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print("\nSegment distribution:")
    print(df["segment_label"].value_counts().sort_index())