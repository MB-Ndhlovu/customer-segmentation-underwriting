"""Synthetic customer dataset for underwriting segmentation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Segment proportions and parameters
    # 0: Mass Market (40%)    — low income, mid credit, short tenure
    # 1: Rising Prime (30%)   — medium income, good credit, growing tenure
    # 2: Established Prime (20%) — high income, excellent credit, long tenure
    # 3: Subprime High-Risk (10%) — low income, poor credit, unstable

    segments = [0, 1, 2, 3]
    weights  = [0.40, 0.30, 0.20, 0.10]

    segment_labels = np.random.choice(segments, size=n, p=weights)

    income_base = {
        0: (45000, 20000),
        1: (75000, 25000),
        2: (140000, 50000),
        3: (30000, 15000),
    }
    credit_base = {
        0: (650, 60),
        1: (720, 50),
        2: (780, 40),
        3: (560, 70),
    }
    employment_base = {
        0: (3, 2),
        1: (6, 3),
        2: (12, 5),
        3: (2, 1.5),
    }
    dti_base = {
        0: (0.30, 0.10),
        1: (0.25, 0.08),
        2: (0.20, 0.06),
        3: (0.45, 0.15),
    }
    loan_count_base = {
        0: (3, 2),
        1: (2, 1.5),
        2: (1, 0.8),
        3: (6, 3),
    }
    age_base = {
        0: (28, 4),
        1: (35, 5),
        2: (45, 7),
        3: (25, 3),
    }
    home_ownership_probs = {
        0: {'rent': 0.7, 'own': 0.2, 'mortgage': 0.1},
        1: {'rent': 0.4, 'own': 0.3, 'mortgage': 0.3},
        2: {'rent': 0.1, 'own': 0.5, 'mortgage': 0.4},
        3: {'rent': 0.85, 'own': 0.1, 'mortgage': 0.05},
    }
    verified_income_probs = {
        0: 0.5,
        1: 0.7,
        2: 0.95,
        3: 0.3,
    }

    records = []
    for seg in segment_labels:
        income   = max(15000, np.random.normal(income_base[seg][0], income_base[seg][1]))
        credit   = max(300, min(850, np.random.normal(credit_base[seg][0], credit_base[seg][1])))
        empl_yrs = max(0, np.random.normal(employment_base[seg][0], employment_base[seg][1]))
        dti      = max(0.05, min(0.80, np.random.normal(dti_base[seg][0], dti_base[seg][1])))
        loans    = max(0, int(np.random.normal(loan_count_base[seg][0], loan_count_base[seg][1])))
        age      = max(18, min(75, np.random.normal(age_base[seg][0], age_base[seg][1])))
        home_options = list(home_ownership_probs[seg].keys())
        home_probs   = list(home_ownership_probs[seg].values())
        home         = np.random.choice(home_options, p=home_probs)
        verified     = np.random.random() < verified_income_probs[seg]

        records.append({
            'income':            round(income, 2),
            'credit_score':      int(round(credit)),
            'employment_years':  round(empl_yrs, 1),
            'debt_to_income':    round(dti, 4),
            'loan_history_count': loans,
            'age':               int(round(age)),
            'home_ownership':    home,
            'verified_income':  verified,
            'segment_label':    seg,
        })

    df = pd.DataFrame(records)

    # Encode home_ownership
    le = LabelEncoder()
    df['home_ownership_enc'] = le.fit_transform(df['home_ownership'])
    df['verified_income_enc'] = df['verified_income'].astype(int)

    return df


if __name__ == '__main__':
    df = generate_customer_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSegment distribution:\n{df['segment_label'].value_counts().sort_index()}")