import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_NAMES = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
SEGMENT_COUNTS = [1800, 1400, 1000, 800]

def load_data():
    segment_params = {
        "Mass Market": dict(
            income_range=(28000, 55000),
            credit_score_range=(580, 680),
            employment_years_range=(1, 8),
            debt_to_income_range=(0.15, 0.40),
            loan_history_range=(0, 4),
            age_range=(22, 45),
            home_ownership_prob=0.25,
            verified_income_prob=0.60,
        ),
        "Rising Prime": dict(
            income_range=(48000, 90000),
            credit_score_range=(660, 760),
            employment_years_range=(3, 12),
            debt_to_income_range=(0.10, 0.30),
            loan_history_range=(1, 6),
            age_range=(26, 40),
            home_ownership_prob=0.45,
            verified_income_prob=0.80,
        ),
        "Established Prime": dict(
            income_range=(75000, 180000),
            credit_score_range=(720, 850),
            employment_years_range=(5, 25),
            debt_to_income_range=(0.05, 0.25),
            loan_history_range=(2, 10),
            age_range=(32, 60),
            home_ownership_prob=0.80,
            verified_income_prob=0.95,
        ),
        "Subprime High-Risk": dict(
            income_range=(15000, 35000),
            credit_score_range=(450, 600),
            employment_years_range=(0, 4),
            debt_to_income_range=(0.35, 0.65),
            loan_history_range=(0, 3),
            age_range=(18, 38),
            home_ownership_prob=0.10,
            verified_income_prob=0.30,
        ),
    }

    records = []
    for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
        p = segment_params[seg_name]
        n = SEGMENT_COUNTS[seg_idx]
        for _ in range(n):
            income = np.random.uniform(*p["income_range"])
            credit_score = int(np.random.uniform(*p["credit_score_range"]))
            employment_years = round(np.random.uniform(*p["employment_years_range"]), 1)
            dti = round(np.random.uniform(*p["debt_to_income_range"]), 3)
            loan_count = int(np.random.randint(*p["loan_history_range"]))
            age = int(np.random.randint(*p["age_range"]))
            home_ownership = 1 if np.random.random() < p["home_ownership_prob"] else 0
            verified_income = 1 if np.random.random() < p["verified_income_prob"] else 0

            records.append({
                "income": income,
                "credit_score": credit_score,
                "employment_years": employment_years,
                "debt_to_income": dti,
                "loan_history_count": loan_count,
                "age": age,
                "home_ownership": home_ownership,
                "verified_income": verified_income,
                "true_segment": seg_name,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_data()
    df.to_csv("/home/workspace/Projects/customer-segmentation-underwriting/customer_data.csv", index=False)
    print(f"Generated {len(df)} rows across {len(SEGMENT_NAMES)} segments.")
    print(df["true_segment"].value_counts().to_string())