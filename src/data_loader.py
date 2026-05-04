import numpy as np
import pandas as pd

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer dataset with 4 distinct underwriting segments.

    Segments are engineered via parameterised distributions:
      0 – Mass Market      : moderate income, average credit, standard employment
      1 – Rising Prime     : growing income, improving credit, short tenure
      2 – Established Prime: high income, excellent credit, long tenure
      3 – Subprime High-Risk: low income, poor credit, high DTI, many loans
    """
    np.random.seed(seed)

    # Cluster sizes roughly 35/25/25/15 to reflect real portfolio distribution
    sizes = [1750, 1250, 1250, 750]
    labels = [0, 1, 2, 3]

    records = []
    for label, size in zip(labels, sizes):
        if label == 0:  # Mass Market
            income = np.random.normal(58000, 12000, size)
            credit_score = np.random.normal(660, 50, size)
            employment_years = np.random.exponential(4.0, size)
            debt_to_income = np.random.beta(3, 7, size) * 0.40
            loan_history_count = np.random.poisson(2, size)
            age = np.random.normal(38, 10, size)
            home_ownership = np.random.binomial(1, 0.45, size)
            verified_income = np.random.binomial(1, 0.60, size)

        elif label == 1:  # Rising Prime
            income = np.random.normal(72000, 18000, size)
            credit_score = np.random.normal(710, 55, size)
            employment_years = np.random.exponential(2.2, size)
            debt_to_income = np.random.beta(2, 9, size) * 0.35
            loan_history_count = np.random.poisson(1, size)
            age = np.random.normal(31, 7, size)
            home_ownership = np.random.binomial(1, 0.30, size)
            verified_income = np.random.binomial(1, 0.50, size)

        elif label == 2:  # Established Prime
            income = np.random.normal(115000, 30000, size)
            credit_score = np.random.normal(780, 45, size)
            employment_years = np.random.exponential(8.0, size)
            debt_to_income = np.random.beta(2, 10, size) * 0.28
            loan_history_count = np.random.poisson(3, size)
            age = np.random.normal(48, 10, size)
            home_ownership = np.random.binomial(1, 0.78, size)
            verified_income = np.random.binomial(1, 0.88, size)

        else:  # label == 3 — Subprime High-Risk
            income = np.random.normal(32000, 9000, size)
            credit_score = np.random.normal(560, 60, size)
            employment_years = np.random.exponential(1.8, size)
            debt_to_income = np.random.beta(5, 5, size) * 0.55
            loan_history_count = np.random.poisson(5, size)
            age = np.random.normal(34, 9, size)
            home_ownership = np.random.binomial(1, 0.20, size)
            verified_income = np.random.binomial(1, 0.35, size)

        for i in range(size):
            records.append(
                {
                    "income": max(15000, income[i]),
                    "credit_score": min(850, max(300, credit_score[i])),
                    "employment_years": max(0, employment_years[i]),
                    "debt_to_income": max(0.0, min(0.9, debt_to_income[i])),
                    "loan_history_count": max(0, loan_history_count[i]),
                    "age": min(80, max(18, age[i])),
                    "home_ownership": int(home_ownership[i]),
                    "verified_income": int(verified_income[i]),
                    "_true_segment": label,
                }
            )

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nTrue segment counts:\n{df['_true_segment'].value_counts().sort_index()}")