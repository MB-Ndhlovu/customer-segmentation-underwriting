"""
Synthetic customer dataset generator for lending/underwriting segmentation.
Generates 5000 rows with features that naturally cluster into 4 distinct segments:
  0 = Mass Market
  1 = Rising Prime
  2 = Established Prime
  3 = Subprime High-Risk
"""

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
    # income, credit_score, employment_years, dti, loan_count, age, home_ownership_weights, verified_prob
    0: dict(  # Mass Market — middle of the road
        income_mean=380000, income_std=90000,
        credit_mean=545, credit_std=65,
        emp_mean=2.5, emp_std=1.8,
        dti_mean=0.28, dti_std=0.07,
        loan_mean=2.5, loan_std=1.2,
        age_mean=34, age_std=7,
        home_weights={"rent": 0.50, "own": 0.15, "mortgage": 0.25, "other": 0.10},
        verified_prob=0.45,
    ),
    1: dict(  # Rising Prime — good but not top tier
        income_mean=620000, income_std=110000,
        credit_mean=665, credit_std=55,
        emp_mean=4.5, emp_std=2.0,
        dti_mean=0.20, dti_std=0.06,
        loan_mean=2.0, loan_std=0.9,
        age_mean=40, age_std=6,
        home_weights={"rent": 0.25, "own": 0.20, "mortgage": 0.42, "other": 0.13},
        verified_prob=0.72,
    ),
    2: dict(  # Established Prime — high earners, stable, low risk
        income_mean=1050000, income_std=220000,
        credit_mean=765, credit_std=48,
        emp_mean=8.5, emp_std=3.5,
        dti_mean=0.13, dti_std=0.04,
        loan_mean=1.2, loan_std=0.7,
        age_mean=48, age_std=8,
        home_weights={"own": 0.55, "mortgage": 0.35, "rent": 0.07, "other": 0.03},
        verified_prob=0.95,
    ),
    3: dict(  # Subprime High-Risk — low income, poor credit, young, high dti
        income_mean=210000, income_std=55000,
        credit_mean=435, credit_std=52,
        emp_mean=1.5, emp_std=1.0,
        dti_mean=0.48, dti_std=0.09,
        loan_mean=4.5, loan_std=1.5,
        age_mean=29, age_std=5,
        home_weights={"rent": 0.72, "other": 0.15, "mortgage": 0.08, "own": 0.05},
        verified_prob=0.22,
    ),
}

SEGMENT_COUNTS = {0: 2000, 1: 1400, 2: 1000, 3: 600}


def generate_dataset(n_total: int = 5000) -> pd.DataFrame:
    records = []
    for seg_id, params in SEGMENT_PARAMS.items():
        n = SEGMENT_COUNTS.get(seg_id, n_total // 4)
        p = params

        incomes = np.random.normal(p["income_mean"], p["income_std"], n).clip(50000, None)
        credit = np.random.normal(p["credit_mean"], p["credit_std"], n).clip(300, 850).astype(int)
        emp = np.random.exponential(p["emp_mean"], n).clip(0.1, 30)
        dti = np.random.normal(p["dti_mean"], p["dti_std"], n).clip(0.01, 0.95)
        loans = np.random.poisson(max(0.5, p["loan_mean"] - 1), n).astype(int) + 1
        ages = np.random.normal(p["age_mean"], p["age_std"], n).clip(18, 80).astype(int)

        home_cats = list(p["home_weights"].keys())
        home_probs = list(p["home_weights"].values())
        home = np.random.choice(home_cats, size=n, p=home_probs)
        verified = (np.random.random(n) < p["verified_prob"]).astype(int)

        for i in range(n):
            records.append({
                "income": round(incomes[i], 2),
                "credit_score": credit[i],
                "employment_years": round(emp[i], 2),
                "debt_to_income": round(dti[i], 4),
                "loan_history_count": loans[i],
                "age": ages[i],
                "home_ownership": home[i],
                "verified_income": verified[i],
                "_segment_label": seg_id,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_data() -> pd.DataFrame:
    df = generate_dataset(5000)
    le = LabelEncoder()
    df["home_ownership_encoded"] = le.fit_transform(df["home_ownership"])
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nSegment distribution:")
    print(df["_segment_label"].value_counts().sort_index())