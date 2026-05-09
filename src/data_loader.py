import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n=5000, seed=42):
    """Generate 5,000 synthetic customer records across 4 distinct segments."""
    np.random.seed(seed)

    # Segment proportions
    n0 = int(n * 0.35)  # Mass Market
    n1 = int(n * 0.25)  # Rising Prime
    n2 = int(n * 0.20)  # Established Prime
    n3 = n - n0 - n1 - n2  # Subprime High-Risk

    segments = [0] * n0 + [1] * n1 + [2] * n2 + [3] * n3
    np.random.shuffle(segments)

    data = []
    for seg in segments:
        if seg == 0:  # Mass Market
            income = np.random.normal(380_000, 80_000)
            credit_score = np.random.normal(620, 60)
            employment_years = np.random.exponential(4.0)
            dti = np.random.beta(4, 6)
            loan_count = np.random.poisson(2)
            age = np.random.normal(35, 8)
            home_owner = np.random.binomial(1, 0.45)
            verified = np.random.binomial(1, 0.7)

        elif seg == 1:  # Rising Prime
            income = np.random.normal(650_000, 120_000)
            credit_score = np.random.normal(700, 50)
            employment_years = np.random.exponential(2.5)
            dti = np.random.beta(3, 8)
            loan_count = np.random.poisson(1)
            age = np.random.normal(29, 5)
            home_owner = np.random.binomial(1, 0.30)
            verified = np.random.binomial(1, 0.80)

        elif seg == 2:  # Established Prime
            income = np.random.normal(950_000, 200_000)
            credit_score = np.random.normal(780, 40)
            employment_years = np.random.exponential(9.0)
            dti = np.random.beta(2, 10)
            loan_count = np.random.poisson(1)
            age = np.random.normal(45, 8)
            home_owner = np.random.binomial(1, 0.80)
            verified = np.random.binomial(1, 0.95)

        else:  # Subprime High-Risk
            income = np.random.normal(180_000, 60_000)
            credit_score = np.random.normal(500, 55)
            employment_years = np.random.exponential(1.5)
            dti = np.random.beta(7, 4)
            loan_count = np.random.poisson(4)
            age = np.random.normal(32, 9)
            home_owner = np.random.binomial(1, 0.15)
            verified = np.random.binomial(1, 0.40)

        data.append({
            "income": max(50_000, income),
            "credit_score": min(850, max(300, credit_score)),
            "employment_years": max(0, employment_years),
            "debt_to_income": min(0.99, max(0.01, dti)),
            "loan_history_count": max(0, loan_count),
            "age": min(75, max(18, age)),
            "home_ownership": home_owner,
            "verified_income": verified,
        })

    df = pd.DataFrame(data)
    df["segment_label"] = segments
    return df


def load_data():
    """Load or generate the customer dataset."""
    return generate_synthetic_data(n=5000)


if __name__ == "__main__":
    df = load_data()
    print(df.describe())
    print("\nSegment distribution:\n", df["segment_label"].value_counts().sort_index())
