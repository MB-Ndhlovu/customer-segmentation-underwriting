"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_customer_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data with realistic underwriting features.

    Produces 4 distinct clusters:
    - Mass Market (label 0): moderate income, decent credit, standard employment
    - Rising Prime (label 1): growing income, improving credit, stable employment
    - Established Prime (label 2): high income, excellent credit, long tenure
    - Subprime High-Risk (label 3): low income, poor credit, short employment
    """
    rng = np.random.default_rng(seed)

    segments = []
    n_per_segment = n_rows // 4

    # Mass Market — moderate risk, largest group
    mass_market = {
        "income": rng.normal(55000, 12000, n_per_segment),
        "credit_score": rng.integers(620, 700, n_per_segment),
        "employment_years": rng.exponential(4, n_per_segment),
        "debt_to_income": rng.normal(0.28, 0.08, n_per_segment),
        "loan_history_count": rng.integers(0, 4, n_per_segment),
        "age": rng.integers(22, 45, n_per_segment),
        "home_ownership": rng.choice([0, 1, 2], n_per_segment, p=[0.4, 0.35, 0.25]),
        "verified_income": rng.choice([0, 1], n_per_segment, p=[0.3, 0.7]),
    }
    segments.append((mass_market, 0))

    # Rising Prime — growing creditworthiness
    rising_prime = {
        "income": rng.normal(75000, 15000, n_per_segment),
        "credit_score": rng.integers(680, 760, n_per_segment),
        "employment_years": rng.exponential(6, n_per_segment),
        "debt_to_income": rng.normal(0.22, 0.06, n_per_segment),
        "loan_history_count": rng.integers(1, 5, n_per_segment),
        "age": rng.integers(25, 50, n_per_segment),
        "home_ownership": rng.choice([0, 1, 2], n_per_segment, p=[0.2, 0.45, 0.35]),
        "verified_income": rng.choice([0, 1], n_per_segment, p=[0.1, 0.9]),
    }
    segments.append((rising_prime, 1))

    # Established Prime — high credit quality
    established_prime = {
        "income": rng.normal(110000, 25000, n_per_segment),
        "credit_score": rng.integers(740, 840, n_per_segment),
        "employment_years": rng.exponential(12, n_per_segment),
        "debt_to_income": rng.normal(0.15, 0.05, n_per_segment),
        "loan_history_count": rng.integers(2, 8, n_per_segment),
        "age": rng.integers(30, 60, n_per_segment),
        "home_ownership": rng.choice([0, 1, 2], n_per_segment, p=[0.05, 0.30, 0.65]),
        "verified_income": rng.choice([0, 1], n_per_segment, p=[0.02, 0.98]),
    }
    segments.append((established_prime, 2))

    # Subprime High-Risk — elevated default risk
    subprime = {
        "income": rng.normal(32000, 8000, n_per_segment),
        "credit_score": rng.integers(500, 619, n_per_segment),
        "employment_years": rng.exponential(1.5, n_per_segment),
        "debt_to_income": rng.normal(0.42, 0.10, n_per_segment),
        "loan_history_count": rng.integers(0, 6, n_per_segment),
        "age": rng.integers(18, 38, n_per_segment),
        "home_ownership": rng.choice([0, 1, 2], n_per_segment, p=[0.65, 0.25, 0.10]),
        "verified_income": rng.choice([0, 1], n_per_segment, p=[0.55, 0.45]),
    }
    segments.append((subprime, 3))

    dfs = []
    for seg_data, label in segments:
        df_seg = pd.DataFrame(seg_data)
        df_seg["segment_label"] = label
        dfs.append(df_seg)

    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Clip outliers to realistic ranges
    df["income"] = df["income"].clip(lower=10000, upper=500000)
    df["credit_score"] = df["credit_score"].clip(lower=300, upper=850)
    df["employment_years"] = df["employment_years"].clip(lower=0, upper=50)
    df["debt_to_income"] = df["debt_to_income"].clip(lower=0.01, upper=0.80)
    df["loan_history_count"] = df["loan_history_count"].clip(lower=0, upper=20)
    df["age"] = df["age"].clip(lower=18, upper=75)

    return df


def get_feature_columns() -> list:
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
    ]


def load_data(n_rows: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Load or generate customer data."""
    df = generate_customer_data(n_rows=n_rows, seed=seed)
    X = df[get_feature_columns()]
    y = df["segment_label"]
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(f"Generated {len(X)} rows with {X.shape[1]} features")
    print(X.describe())
    print("\nSegment distribution:")
    print(y.value_counts().sort_index())