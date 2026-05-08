import numpy as np
import pandas as pd

def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)

    # Segment distribution (approximate, will refine through clustering)
    # We generate with known structures so clusters emerge naturally
    segment_probs = [0.30, 0.30, 0.25, 0.15]

    # Allocate segment to each row
    segment_assignments = np.random.choice(4, size=n, p=segment_probs)

    data = {
        'income': np.zeros(n),
        'credit_score': np.zeros(n),
        'employment_years': np.zeros(n),
        'debt_to_income': np.zeros(n),
        'loan_history_count': np.zeros(n),
        'age': np.zeros(n),
        'home_ownership_status': np.zeros(n),  # 0=rent, 1=mortgage, 2=own
        'verified_income': np.zeros(n),         # 0=no, 1=yes
        'segment_label': segment_assignments
    }

    for i in range(n):
        seg = segment_assignments[i]

        # Mass Market (0): low income, mid credit, short history
        if seg == 0:
            data['income'][i] = np.random.normal(35000, 8000)
            data['credit_score'][i] = np.random.normal(620, 50)
            data['employment_years'][i] = np.random.exponential(2)
            data['debt_to_income'][i] = np.random.normal(0.28, 0.08)
            data['loan_history_count'][i] = np.random.poisson(1.5)
            data['age'][i] = np.random.randint(22, 45)
            data['home_ownership_status'][i] = np.random.choice([0, 1], p=[0.8, 0.2])
            data['verified_income'][i] = np.random.choice([0, 1], p=[0.7, 0.3])

        # Rising Prime (1): moderate income, improving credit
        elif seg == 1:
            data['income'][i] = np.random.normal(65000, 12000)
            data['credit_score'][i] = np.random.normal(700, 40)
            data['employment_years'][i] = np.random.exponential(4)
            data['debt_to_income'][i] = np.random.normal(0.22, 0.06)
            data['loan_history_count'][i] = np.random.poisson(2)
            data['age'][i] = np.random.randint(28, 50)
            data['home_ownership_status'][i] = np.random.choice([0, 1, 2], p=[0.4, 0.45, 0.15])
            data['verified_income'][i] = np.random.choice([0, 1], p=[0.4, 0.6])

        # Established Prime (2): high income, strong credit, stable
        elif seg == 2:
            data['income'][i] = np.random.normal(110000, 25000)
            data['credit_score'][i] = np.random.normal(760, 35)
            data['employment_years'][i] = np.random.exponential(8)
            data['debt_to_income'][i] = np.random.normal(0.15, 0.05)
            data['loan_history_count'][i] = np.random.poisson(3)
            data['age'][i] = np.random.randint(35, 60)
            data['home_ownership_status'][i] = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
            data['verified_income'][i] = np.random.choice([0, 1], p=[0.1, 0.9])

        # Subprime High-Risk (3): low credit, high DTI, many prior loans
        elif seg == 3:
            data['income'][i] = np.random.normal(28000, 7000)
            data['credit_score'][i] = np.random.normal(560, 45)
            data['employment_years'][i] = np.random.exponential(1.5)
            data['debt_to_income'][i] = np.random.normal(0.42, 0.10)
            data['loan_history_count'][i] = np.random.poisson(4)
            data['age'][i] = np.random.randint(20, 40)
            data['home_ownership_status'][i] = np.random.choice([0, 1], p=[0.9, 0.1])
            data['verified_income'][i] = np.random.choice([0, 1], p=[0.85, 0.15])

    df = pd.DataFrame(data)

    # Clip to realistic bounds
    df['income'] = df['income'].clip(15000, 300000)
    df['credit_score'] = df['credit_score'].clip(450, 850)
    df['employment_years'] = df['employment_years'].clip(0, 40)
    df['debt_to_income'] = df['debt_to_income'].clip(0.01, 0.60)
    df['loan_history_count'] = df['loan_history_count'].clip(0, 15)
    df['age'] = df['age'].clip(18, 70)

    return df


def load_data(path=None):
    if path:
        return pd.read_csv(path)
    return generate_synthetic_data()