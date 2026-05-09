"""Generate 5000 synthetic customer records for underwriting segmentation."""

import numpy as np
import pandas as pd

SEGMENT_PARAMS = {
    0: {  # Mass Market
        "income": ("uniform", 35000, 80000),
        "credit_score": ("uniform", 580, 700),
        "employment_years": ("uniform", 1, 10),
        "debt_to_income": ("uniform", 0.15, 0.35),
        "loan_history_count": ("uniform", 0, 4),
        "age": ("uniform", 22, 45),
        "home_ownership": ("categorical", [("rent", 0.7), ("own", 0.2), ("mortgage", 0.1)]),
        "verified_income": ("categorical", [("unverified", 0.6), ("verified", 0.4)]),
        "weight": 0.35,
    },
    1: {  # Rising Prime
        "income": ("uniform", 60000, 110000),
        "credit_score": ("uniform", 680, 760),
        "employment_years": ("uniform", 3, 12),
        "debt_to_income": ("uniform", 0.10, 0.28),
        "loan_history_count": ("uniform", 1, 5),
        "age": ("uniform", 26, 40),
        "home_ownership": ("categorical", [("rent", 0.4), ("own", 0.35), ("mortgage", 0.25)]),
        "verified_income": ("categorical", [("unverified", 0.3), ("verified", 0.7)]),
        "weight": 0.30,
    },
    2: {  # Established Prime
        "income": ("uniform", 90000, 200000),
        "credit_score": ("uniform", 730, 850),
        "employment_years": ("uniform", 5, 25),
        "debt_to_income": ("uniform", 0.05, 0.22),
        "loan_history_count": ("uniform", 2, 8),
        "age": ("uniform", 32, 60),
        "home_ownership": ("categorical", [("rent", 0.1), ("own", 0.5), ("mortgage", 0.4)]),
        "verified_income": ("categorical", [("unverified", 0.1), ("verified", 0.9)]),
        "weight": 0.20,
    },
    3: {  # Subprime High-Risk
        "income": ("uniform", 18000, 45000),
        "credit_score": ("uniform", 450, 620),
        "employment_years": ("uniform", 0, 4),
        "debt_to_income": ("uniform", 0.30, 0.55),
        "loan_history_count": ("uniform", 0, 6),
        "age": ("uniform", 18, 35),
        "home_ownership": ("categorical", [("rent", 0.8), ("own", 0.1), ("mortgage", 0.1)]),
        "verified_income": ("categorical", [("unverified", 0.85), ("verified", 0.15)]),
        "weight": 0.15,
    },
}


def _sample(param_spec):
    dist = param_spec[0]
    if dist == "uniform":
        return np.random.uniform(param_spec[1], param_spec[2])
    elif dist == "categorical":
        categories = [item[0] for item in param_spec[1]]
        probs = [item[1] for item in param_spec[1]]
        return np.random.choice(categories, p=probs)
    raise ValueError(f"Unknown distribution: {dist}")


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    weights = [SEGMENT_PARAMS[s]["weight"] for s in sorted(SEGMENT_PARAMS)]
    segment_ids = np.random.choice([0, 1, 2, 3], size=n, p=weights)

    rows = []
    for seg in segment_ids:
        params = SEGMENT_PARAMS[seg]
        row = {"segment_label": seg}
        for feat, spec in params.items():
            if feat == "weight":
                continue
            row[feat] = _sample(spec)
        rows.append(row)

    df = pd.DataFrame(rows)

    home_map = {"rent": 0, "own": 1, "mortgage": 2}
    income_verif_map = {"unverified": 0, "verified": 1}
    df["home_ownership"] = df["home_ownership"].map(home_map)
    df["verified_income"] = df["verified_income"].map(income_verif_map)

    df["income"] = df["income"].round(2)
    df["debt_to_income"] = df["debt_to_income"].round(4)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print("\nSegment distribution:")
    print(df["segment_label"].value_counts().sort_index())