import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer data for underwriting segmentation."""

    segments = {
        "Mass Market": 0,
        "Rising Prime": 1,
        "Established Prime": 2,
        "Subprime High-Risk": 3,
    }

    data = []
    for _ in range(n):
        seg = np.random.choice(list(segments.keys()), p=[0.30, 0.25, 0.25, 0.20])

        if seg == "Mass Market":
            income = np.random.normal(55000, 12000)
            credit_score = np.random.randint(620, 700)
            employment_years = np.random.randint(2, 10)
            debt_to_income = np.random.uniform(0.20, 0.38)
            loan_history_count = np.random.randint(1, 4)
            age = np.random.randint(25, 50)
            home_ownership = np.random.choice(["rent", "own", "rent", "own"], p=[0.5, 0.3, 0.2, 0.0])
            verified_income = np.random.choice([True, False], p=[0.65, 0.35])

        elif seg == "Rising Prime":
            income = np.random.normal(75000, 18000)
            credit_score = np.random.randint(680, 760)
            employment_years = np.random.randint(1, 5)
            debt_to_income = np.random.uniform(0.15, 0.30)
            loan_history_count = np.random.randint(0, 3)
            age = np.random.randint(23, 38)
            home_ownership = np.random.choice(["rent", "rent", "own"], p=[0.6, 0.25, 0.15])
            verified_income = np.random.choice([True, False], p=[0.80, 0.20])

        elif seg == "Established Prime":
            income = np.random.normal(130000, 35000)
            credit_score = np.random.randint(740, 850)
            employment_years = np.random.randint(8, 30)
            debt_to_income = np.random.uniform(0.08, 0.22)
            loan_history_count = np.random.randint(2, 7)
            age = np.random.randint(35, 60)
            home_ownership = np.random.choice(["own", "own", "mortgage"], p=[0.50, 0.30, 0.20])
            verified_income = np.random.choice([True], p=[1.0])

        else:  # Subprime High-Risk
            income = np.random.normal(28000, 9000)
            credit_score = np.random.randint(500, 640)
            employment_years = np.random.randint(0, 4)
            debt_to_income = np.random.uniform(0.35, 0.60)
            loan_history_count = np.random.randint(3, 9)
            age = np.random.randint(20, 45)
            home_ownership = np.random.choice(["rent", "rent", "own"], p=[0.75, 0.20, 0.05])
            verified_income = np.random.choice([True, False], p=[0.40, 0.60])

        data.append({
            "income": max(15000, income),
            "credit_score": min(max(300, credit_score), 850),
            "employment_years": max(0, employment_years),
            "debt_to_income": min(max(0.0, debt_to_income), 0.95),
            "loan_history_count": max(0, loan_history_count),
            "age": min(max(18, age), 80),
            "home_ownership": home_ownership,
            "verified_income": verified_income,
            "true_segment": segments[seg],
        })

    df = pd.DataFrame(data)

    home_map = {"rent": 0, "own": 1, "mortgage": 2}
    df["home_ownership"] = df["home_ownership"].map(home_map)
    df["verified_income"] = df["verified_income"].astype(int)

    return df


if __name__ == "__main__":
    df = generate_customer_data(5000)
    print(df.head())
    print(df.describe())
    print(df["true_segment"].value_counts().sort_index())