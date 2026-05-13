import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_PARAMS = {
    0: {  # Mass Market
        "income": (35000, 8000),
        "credit_score": (580, 60),
        "employment_years": (3, 2),
        "debt_to_income": (0.25, 0.10),
        "loan_history_count": (2, 1.5),
        "age": (28, 6),
        "home_ownership": 0.30,
        "verified_income": 0.40,
        "weight": 0.35,
    },
    1: {  # Rising Prime
        "income": (95000, 15000),
        "credit_score": (680, 50),
        "employment_years": (6, 3),
        "debt_to_income": (0.20, 0.08),
        "loan_history_count": (4, 2),
        "age": (35, 7),
        "home_ownership": 0.55,
        "verified_income": 0.75,
        "weight": 0.30,
    },
    2: {  # Established Prime
        "income": (160000, 40000),
        "credit_score": (760, 40),
        "employment_years": (15, 5),
        "debt_to_income": (0.15, 0.06),
        "loan_history_count": (8, 3),
        "age": (48, 8),
        "home_ownership": 0.85,
        "verified_income": 0.95,
        "weight": 0.20,
    },
    3: {  # Subprime High-Risk
        "income": (28000, 6000),
        "credit_score": (520, 45),
        "employment_years": (2, 1.5),
        "debt_to_income": (0.45, 0.12),
        "loan_history_count": (6, 3),
        "age": (30, 8),
        "home_ownership": 0.15,
        "verified_income": 0.20,
        "weight": 0.15,
    },
}

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    weights = [SEGMENT_PARAMS[k]["weight"] for k in range(4)]
    segment_labels = np.random.choice([0, 1, 2, 3], size=n, p=weights)

    rows = []
    for seg in range(4):
        mask = segment_labels == seg
        count = mask.sum()
        p = SEGMENT_PARAMS[seg]

        income = np.clip(np.random.normal(p["income"][0], p["income"][1], count), 15000, 500000).astype(int)
        credit_score = np.clip(np.random.normal(p["credit_score"][0], p["credit_score"][1], count), 300, 850).astype(int)
        employment_years = np.clip(np.random.normal(p["employment_years"][0], p["employment_years"][1], count), 0, 50).astype(int)
        debt_to_income = np.clip(np.random.normal(p["debt_to_income"][0], p["debt_to_income"][1], count), 0.0, 1.0).round(4)
        loan_history_count = np.clip(np.random.poisson(max(1, p["loan_history_count"][0] - 1), count), 0, 30).astype(int)
        age = np.clip(np.random.normal(p["age"][0], p["age"][1], count), 18, 80).astype(int)
        home_ownership = (np.random.rand(count) < p["home_ownership"]).astype(int)
        verified_income = (np.random.rand(count) < p["verified_income"]).astype(int)

        df_seg = pd.DataFrame({
            "income": income,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "loan_history_count": loan_history_count,
            "age": age,
            "home_ownership": home_ownership,
            "verified_income": verified_income,
            "segment_label": seg,
        })
        rows.append(df_seg)

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_customer_data(5000)
    print(df.head())
    print(df["segment_label"].value_counts().sort_index())
