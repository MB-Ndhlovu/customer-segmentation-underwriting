"""Generate synthetic customer dataset for underwriting segmentation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

SEGMENTS = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def generate_segment_params(segment_label: int) -> dict:
    """Return base distribution parameters for each segment."""
    params = {
        0: dict(income_mean=55000, income_std=18000, credit_mean=645, credit_std=55,
                employment_mean=5, employment_std=3, dti_mean=0.28, dti_std=0.10,
                loans_mean=2, loans_std=1.5, age_mean=34, age_std=8,
                home_own_prob=0.35, verified_prob=0.55),
        1: dict(income_mean=72000, income_std=22000, credit_mean=715, credit_std=48,
                employment_mean=7, employment_std=3.5, dti_mean=0.22, dti_std=0.09,
                loans_mean=1.5, loans_std=1.2, age_mean=38, age_std=7,
                home_own_prob=0.50, verified_prob=0.72),
        2: dict(income_mean=115000, income_std=35000, credit_mean=778, credit_std=42,
                employment_mean=12, employment_std=5, dti_mean=0.18, dti_std=0.08,
                loans_mean=2.5, loans_std=2, age_mean=45, age_std=9,
                home_own_prob=0.72, verified_prob=0.88),
        3: dict(income_mean=28000, income_std=12000, credit_mean=572, credit_std=62,
                employment_mean=3, employment_std=2.5, dti_mean=0.40, dti_std=0.14,
                loans_mean=4, loans_std=2, age_mean=29, age_std=7,
                home_own_prob=0.18, verified_prob=0.30),
    }
    return params[segment_label]


def generate_customers(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic customer records distributed across 4 segments."""
    records = []
    counts = {0: int(n * 0.30), 1: int(n * 0.25), 2: int(n * 0.20), 3: int(n * 0.25)}
    counts[0] += n - sum(counts.values())

    for seg_label, seg_size in counts.items():
        p = generate_segment_params(seg_label)
        for _ in range(seg_size):
            income = max(15000, np.random.normal(p["income_mean"], p["income_std"]))
            credit_score = int(np.clip(np.random.normal(p["credit_mean"], p["credit_std"]), 300, 850))
            employment_years = max(0, np.random.normal(p["employment_mean"], p["employment_std"]))
            debt_to_income = max(0.01, np.random.normal(p["dti_mean"], p["dti_std"]))
            loan_history_count = max(0, int(np.random.poisson(max(0.5, p["loans_mean"]))))
            age = int(np.clip(np.random.normal(p["age_mean"], p["age_std"]), 18, 80))
            home_ownership = "own" if np.random.random() < p["home_own_prob"] else "rent"
            verified_income = "verified" if np.random.random() < p["verified_prob"] else "unverified"

            records.append({
                "segment_label": seg_label,
                "income": round(income, 2),
                "credit_score": credit_score,
                "employment_years": round(employment_years, 2),
                "debt_to_income": round(debt_to_income, 4),
                "loan_history_count": loan_history_count,
                "age": age,
                "home_ownership": home_ownership,
                "verified_income": verified_income,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    le_home = LabelEncoder().fit(["rent", "own"])
    le_verified = LabelEncoder().fit(["unverified", "verified"])
    df["home_ownership_enc"] = le_home.transform(df["home_ownership"])
    df["verified_income_enc"] = le_verified.transform(df["verified_income"])

    return df


if __name__ == "__main__":
    df = generate_customers(5000)
    print(df.head())
    print(f"\nSegment distribution:\n{df['segment_label'].value_counts().sort_index()}")
    df.to_csv("/home/workspace/Projects/customer-segmentation-underwriting/customers.csv", index=False)
    print("\nSaved to customers.csv")