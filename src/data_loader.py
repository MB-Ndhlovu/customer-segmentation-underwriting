"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd

# Segment definitions: (count, income_range, credit_range, emp_years_range, dti_range, loan_count_range, age_range, home_own_prob, verified_income_prob)
SEGMENTS = {
    0: {  # Mass Market
        "count": 1600,
        "income": (35000, 85000),
        "credit_score": (620, 720),
        "employment_years": (1, 8),
        "debt_to_income": (0.15, 0.38),
        "loan_history_count": (1, 5),
        "age": (24, 45),
        "home_ownership": 0.25,
        "verified_income": 0.55,
    },
    1: {  # Rising Prime
        "count": 1400,
        "income": (70000, 140000),
        "credit_score": (700, 800),
        "employment_years": (3, 12),
        "debt_to_income": (0.10, 0.30),
        "loan_history_count": (1, 4),
        "age": (28, 50),
        "home_ownership": 0.55,
        "verified_income": 0.80,
    },
    2: {  # Established Prime
        "count": 1200,
        "income": (110000, 250000),
        "credit_score": (760, 850),
        "employment_years": (8, 25),
        "debt_to_income": (0.05, 0.22),
        "loan_history_count": (0, 3),
        "age": (35, 60),
        "home_ownership": 0.85,
        "verified_income": 0.95,
    },
    3: {  # Subprime High-Risk
        "count": 800,
        "income": (18000, 45000),
        "credit_score": (480, 620),
        "employment_years": (0, 3),
        "debt_to_income": (0.35, 0.65),
        "loan_history_count": (3, 10),
        "age": (20, 40),
        "home_ownership": 0.08,
        "verified_income": 0.25,
    },
}

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

rng = np.random.default_rng(seed=42)


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def generate(n_total: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    rows = []

    # Distribute counts proportionally but fix to get exactly n_total
    segment_counts = {k: v["count"] for k, v in SEGMENTS.items()}
    # Slight random jitter so sum == n_total
    scale = n_total / sum(segment_counts.values())
    counts = {k: max(1, int(v * scale)) for k, v in segment_counts.items()}
    # Correct rounding error
    diff = n_total - sum(counts.values())
    counts[0] += diff

    for seg_id, params in SEGMENTS.items():
        n = counts[seg_id]
        income = rng.integers(params["income"][0], params["income"][1] + 1, size=n)
        credit = rng.integers(params["credit_score"][0], params["credit_score"][1] + 1, size=n)
        emp = rng.uniform(params["employment_years"][0], params["employment_years"][1], size=n)
        dti = rng.uniform(params["debt_to_income"][0], params["debt_to_income"][1], size=n)
        loans = rng.integers(params["loan_history_count"][0], params["loan_history_count"][1] + 1, size=n)
        age = rng.integers(params["age"][0], params["age"][1] + 1, size=n)
        home_own = rng.binomial(1, params["home_ownership"], size=n)
        verified = rng.binomial(1, params["verified_income"], size=n)

        for i in range(n):
            rows.append(
                {
                    "income": int(income[i]),
                    "credit_score": int(credit[i]),
                    "employment_years": round(emp[i], 2),
                    "debt_to_income": round(dti[i], 4),
                    "loan_history_count": int(loans[i]),
                    "age": int(age[i]),
                    "home_ownership": int(home_own[i]),
                    "verified_income": int(verified[i]),
                    "segment_label": seg_id,
                    "segment_name": SEGMENT_NAMES[seg_id],
                }
            )

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print("\nSegment distribution:")
    print(df["segment_label"].value_counts().sort_index())