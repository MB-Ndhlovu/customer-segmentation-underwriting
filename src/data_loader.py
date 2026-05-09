import numpy as np
import pandas as pd

def generate_customer_data(n=5000, seed=42):
    np.random.seed(seed)

    # Segment proportions and parameters
    # Segment 0: Mass Market (40%)
    # Segment 1: Rising Prime (30%)
    # Segment 2: Established Prime (20%)
    # Segment 3: Subprime High-Risk (10%)

    segment_params = {
        0: {  # Mass Market
            'income_range': (25000, 65000),
            'credit_score_range': (620, 720),
            'employment_years_range': (1, 10),
            'debt_to_income_range': (0.15, 0.35),
            'loan_history_range': (0, 3),
            'age_range': (22, 45),
            'home_ownership_probs': (0.3, 0.5, 0.2),  # rent, own, mortgage
            'verified_income_prob': 0.55,
        },
        1: {  # Rising Prime
            'income_range': (55000, 95000),
            'credit_score_range': (680, 760),
            'employment_years_range': (3, 12),
            'debt_to_income_range': (0.20, 0.40),
            'loan_history_range': (1, 5),
            'age_range': (28, 50),
            'home_ownership_probs': (0.25, 0.35, 0.40),
            'verified_income_prob': 0.75,
        },
        2: {  # Established Prime
            'income_range': (80000, 200000),
            'credit_score_range': (740, 850),
            'employment_years_range': (5, 25),
            'debt_to_income_range': (0.10, 0.30),
            'loan_history_range': (2, 8),
            'age_range': (35, 60),
            'home_ownership_probs': (0.10, 0.50, 0.40),
            'verified_income_prob': 0.90,
        },
        3: {  # Subprime High-Risk
            'income_range': (18000, 45000),
            'credit_score_range': (500, 640),
            'employment_years_range': (0, 5),
            'debt_to_income_range': (0.35, 0.65),
            'loan_history_range': (2, 10),
            'age_range': (20, 40),
            'home_ownership_probs': (0.60, 0.15, 0.25),
            'verified_income_prob': 0.30,
        },
    }

    segment_sizes = {0: 2000, 1: 1500, 2: 1000, 3: 500}
    rows = []

    for seg_id, params in segment_params.items():
        n_seg = segment_sizes[seg_id]
        for _ in range(n_seg):
            income = np.random.uniform(*params['income_range'])
            credit_score = int(np.random.uniform(*params['credit_score_range']))
            employment_years = np.random.uniform(*params['employment_years_range'])
            debt_to_income = np.random.uniform(*params['debt_to_income_range'])
            loan_history_count = int(np.random.uniform(*params['loan_history_range']))
            age = int(np.random.uniform(*params['age_range']))

            home_ownership_cat = np.random.choice(
                ['rent', 'own', 'mortgage'],
                p=params['home_ownership_probs']
            )
            verified_income = 1 if np.random.random() < params['verified_income_prob'] else 0

            rows.append({
                'income': round(income, 2),
                'credit_score': credit_score,
                'employment_years': round(employment_years, 2),
                'debt_to_income': round(debt_to_income, 4),
                'loan_history_count': loan_history_count,
                'age': age,
                'home_ownership': home_ownership_cat,
                'verified_income': verified_income,
                'segment_label': seg_id,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def load_data():
    return generate_customer_data(n=5000, seed=42)