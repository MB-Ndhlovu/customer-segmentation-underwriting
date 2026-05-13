"""Generate synthetic customer dataset for underwriting segmentation."""

import numpy as np
import pandas as pd
from typing import Optional

np.random.seed(42)

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def generate_income(segment: int) -> float:
    base = {
        0: 35_000,
        1: 65_000,
        2: 120_000,
        3: 22_000,
    }[segment]
    spread = base * 0.30
    return max(15_000, np.random.normal(base, spread))


def generate_credit_score(segment: int) -> int:
    base = {
        0: 660,
        1: 740,
        2: 800,
        3: 540,
    }[segment]
    return int(np.clip(np.random.normal(base, 25), 300, 850))


def generate_employment_years(segment: int) -> float:
    base = {
        0: 3.0,
        1: 5.0,
        2: 15.0,
        3: 1.5,
    }[segment]
    return max(0.0, np.random.normal(base, 2.5))


def generate_dti(segment: int) -> float:
    base = {
        0: 0.28,
        1: 0.22,
        2: 0.15,
        3: 0.45,
    }[segment]
    return float(np.clip(np.random.normal(base, 0.06), 0.01, 0.80))


def generate_loan_count(segment: int) -> int:
    base = {
        0: 2,
        1: 3,
        2: 4,
        3: 5,
    }[segment]
    return max(0, int(np.random.normal(base, 1.5)))


def generate_age(segment: int) -> int:
    base = {
        0: 30,
        1: 38,
        2: 50,
        3: 27,
    }[segment]
    return int(np.clip(np.random.normal(base, 7), 18, 80))


def generate_homeownership(segment: int) -> str:
    probs = {
        0: [0.30, 0.30, 0.40],
        1: [0.20, 0.30, 0.50],
        2: [0.10, 0.20, 0.70],
        3: [0.55, 0.25, 0.20],
    }
    options = ["rent", "mortgage", "own"]
    return np.random.choice(options, p=probs[segment])


def generate_verified_income(segment: int) -> bool:
    probs = {
        0: 0.50,
        1: 0.78,
        2: 0.92,
        3: 0.25,
    }
    return np.random.random() < probs[segment]


def load_customer_data(n_rows: int = 5000, with_labels: bool = True) -> pd.DataFrame:
    """Generate n_rows synthetic customer records."""
    n_each = n_rows // 4
    rows = []
    for seg in range(4):
        for _ in range(n_each):
            rows.append(
                {
                    "income": generate_income(seg),
                    "credit_score": generate_credit_score(seg),
                    "employment_years": generate_employment_years(seg),
                    "debt_to_income": generate_dti(seg),
                    "loan_history_count": generate_loan_count(seg),
                    "age": generate_age(seg),
                    "home_ownership": generate_homeownership(seg),
                    "verified_income": generate_verified_income(seg),
                }
            )

    df = pd.DataFrame(rows)

    if with_labels:
        # We don't expose true segment to downstream — kept internal for evaluation
        pass

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_customer_data()
    print(df.describe())
    print(df["home_ownership"].value_counts())