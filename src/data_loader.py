import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_PARAMS = {
    0: {"income": (35000, 70000), "credit_score": (620, 720), "employment_years": (1, 8),
        "debt_to_income": (0.15, 0.35), "loan_history_count": (0, 4), "age": (22, 45),
        "home_ownership": 0.30, "verified_income": 0.40},
    1: {"income": (60000, 110000), "credit_score": (680, 780), "employment_years": (3, 12),
        "debt_to_income": (0.10, 0.28), "loan_history_count": (1, 5), "age": (26, 50),
        "home_ownership": 0.55, "verified_income": 0.70},
    2: {"income": (90000, 200000), "credit_score": (740, 850), "employment_years": (5, 25),
        "debt_to_income": (0.05, 0.22), "loan_history_count": (2, 8), "age": (32, 60),
        "home_ownership": 0.85, "verified_income": 0.90},
    3: {"income": (20000, 45000), "credit_score": (500, 640), "employment_years": (0, 4),
        "debt_to_income": (0.30, 0.60), "loan_history_count": (2, 10), "age": (20, 40),
        "home_ownership": 0.10, "verified_income": 0.20},
}

SEGMENT_WEIGHTS = [0.28, 0.25, 0.22, 0.25]

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def generate_customer_data(n=5000):
    records = []
    segment_counts = np.random.multinomial(n, SEGMENT_WEIGHTS)
    for seg_id, count in enumerate(segment_counts):
        p = SEGMENT_PARAMS[seg_id]
        income = np.random.uniform(*p["income"], count)
        credit = np.random.normal(
            (p["credit_score"][0] + p["credit_score"][1]) / 2,
            (p["credit_score"][1] - p["credit_score"][0]) / 4,
            count
        ).clip(p["credit_score"][0], p["credit_score"][1])
        emp_years = np.random.uniform(*p["employment_years"], count).clip(0)
        dti = np.random.uniform(*p["debt_to_income"], count).clip(0, 1)
        loan_count = np.random.randint(p["loan_history_count"][0], p["loan_history_count"][1] + 1, count)
        age = np.random.randint(p["age"][0], p["age"][1] + 1, count)
        home_owned = np.random.random(count) < p["home_ownership"]
        income_verified = np.random.random(count) < p["verified_income"]
        for i in range(count):
            records.append({
                "segment_id": seg_id,
                "income": round(income[i], 2),
                "credit_score": round(credit[i]),
                "employment_years": round(emp_years[i], 2),
                "debt_to_income": round(dti[i], 4),
                "loan_history_count": int(loan_count[i]),
                "age": int(age[i]),
                "home_ownership": int(home_owned[i]),
                "verified_income": int(income_verified[i]),
            })
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def get_segment_name(label):
    return SEGMENT_NAMES.get(label, f"Segment {label}")