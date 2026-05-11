import numpy as np
import pandas as pd


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer dataset with realistic financial profiles.

    Segments are embedded in the data generation process:
    - Segment 0: Mass Market  (avg income ~50k, credit ~650)
    - Segment 1: Rising Prime (avg income ~70k, credit ~700, growing employment)
    - Segment 2: Established Prime (avg income ~120k, credit ~780, long tenure)
    - Segment 3: Subprime High-Risk (avg income ~25k, credit ~580, high DTI)
    """
    np.random.seed(seed)

    # Allocate segment proportions: 40% Mass Market, 25% Rising Prime,
    # 20% Established Prime, 15% Subprime High-Risk
    segment_sizes = [2000, 1250, 1000, 750]
    labels = []
    for seg_id, size in enumerate(segment_sizes):
        labels.extend([seg_id] * size)
    labels = np.array(labels)
    np.random.shuffle(labels)

    income = np.zeros(n)
    credit_score = np.zeros(n)
    employment_years = np.zeros(n)
    debt_to_income = np.zeros(n)
    loan_history_count = np.zeros(n)
    age = np.zeros(n)
    home_ownership_status = np.zeros(n)   # 1=owner, 0=renter
    verified_income = np.zeros(n)         # 1=yes, 0=no

    # Segment 0: Mass Market
    mask0 = labels == 0
    income[mask0] = np.random.normal(50000, 12000, mask0.sum())
    credit_score[mask0] = np.random.normal(650, 50, mask0.sum())
    employment_years[mask0] = np.random.normal(5, 3, mask0.sum())
    debt_to_income[mask0] = np.random.normal(0.28, 0.10, mask0.sum())
    loan_history_count[mask0] = np.random.poisson(2, mask0.sum())
    age[mask0] = np.random.normal(34, 8, mask0.sum())
    home_ownership_status[mask0] = np.random.choice([0, 1], size=mask0.sum(), p=[0.7, 0.3])
    verified_income[mask0] = np.random.choice([0, 1], size=mask0.sum(), p=[0.6, 0.4])

    # Segment 1: Rising Prime
    mask1 = labels == 1
    income[mask1] = np.random.normal(70000, 15000, mask1.sum())
    credit_score[mask1] = np.random.normal(710, 45, mask1.sum())
    employment_years[mask1] = np.random.normal(6, 3, mask1.sum())
    debt_to_income[mask1] = np.random.normal(0.22, 0.08, mask1.sum())
    loan_history_count[mask1] = np.random.poisson(3, mask1.sum())
    age[mask1] = np.random.normal(30, 5, mask1.sum())
    home_ownership_status[mask1] = np.random.choice([0, 1], size=mask1.sum(), p=[0.55, 0.45])
    verified_income[mask1] = np.random.choice([0, 1], size=mask1.sum(), p=[0.3, 0.7])

    # Segment 2: Established Prime
    mask2 = labels == 2
    income[mask2] = np.random.normal(120000, 30000, mask2.sum())
    credit_score[mask2] = np.random.normal(780, 40, mask2.sum())
    employment_years[mask2] = np.random.normal(15, 5, mask2.sum())
    debt_to_income[mask2] = np.random.normal(0.15, 0.06, mask2.sum())
    loan_history_count[mask2] = np.random.poisson(4, mask2.sum())
    age[mask2] = np.random.normal(45, 8, mask2.sum())
    home_ownership_status[mask2] = np.random.choice([0, 1], size=mask2.sum(), p=[0.2, 0.8])
    verified_income[mask2] = np.random.choice([0, 1], size=mask2.sum(), p=[0.1, 0.9])

    # Segment 3: Subprime High-Risk
    mask3 = labels == 3
    income[mask3] = np.random.normal(25000, 8000, mask3.sum())
    credit_score[mask3] = np.random.normal(580, 55, mask3.sum())
    employment_years[mask3] = np.random.normal(2, 1.5, mask3.sum())
    debt_to_income[mask3] = np.random.normal(0.45, 0.12, mask3.sum())
    loan_history_count[mask3] = np.random.poisson(4, mask3.sum())
    age[mask3] = np.random.normal(28, 6, mask3.sum())
    home_ownership_status[mask3] = np.random.choice([0, 1], size=mask3.sum(), p=[0.85, 0.15])
    verified_income[mask3] = np.random.choice([0, 1], size=mask3.sum(), p=[0.8, 0.2])

    # Clamp to realistic bounds
    income = np.clip(income, 10000, 500000)
    credit_score = np.clip(credit_score, 300, 850)
    employment_years = np.clip(employment_years, 0, 50)
    debt_to_income = np.clip(debt_to_income, 0.0, 0.95)
    loan_history_count = np.clip(loan_history_count, 0, 20)
    age = np.clip(age, 18, 80)
    home_ownership_status = home_ownership_status.astype(int)
    verified_income = verified_income.astype(int)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership_status": home_ownership_status,
        "verified_income": verified_income,
    })

    # Add the ground-truth label for evaluation only (not used in training)
    df["segment_label"] = labels

    return df


FEATURE_COLS = [
    "income", "credit_score", "employment_years",
    "debt_to_income", "loan_history_count", "age",
    "home_ownership_status", "verified_income",
]
SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}