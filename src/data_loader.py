import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_PARAMS = {
    0: {  # Mass Market
        'income': (45000, 15000),
        'credit_score': (620, 50),
        'employment_years': (4, 2),
        'debt_to_income': (0.28, 0.08),
        'loan_history_count': (3, 2),
        'age': (32, 8),
        'home_ownership': [0.3, 0.4, 0.3],  # Own, Rent, Mortgage
        'verified_income': 0.5,
    },
    1: {  # Rising Prime
        'income': (72000, 18000),
        'credit_score': (700, 45),
        'employment_years': (6, 3),
        'debt_to_income': (0.22, 0.06),
        'loan_history_count': (4, 2),
        'age': (36, 7),
        'home_ownership': [0.45, 0.3, 0.25],
        'verified_income': 0.75,
    },
    2: {  # Established Prime
        'income': (110000, 30000),
        'credit_score': (760, 40),
        'employment_years': (10, 4),
        'debt_to_income': (0.18, 0.05),
        'loan_history_count': (6, 2),
        'age': (45, 10),
        'home_ownership': [0.7, 0.15, 0.15],
        'verified_income': 0.9,
    },
    3: {  # Subprime High-Risk
        'income': (28000, 8000),
        'credit_score': (540, 40),
        'employment_years': (2, 1.5),
        'debt_to_income': (0.42, 0.10),
        'loan_history_count': (5, 3),
        'age': (28, 6),
        'home_ownership': [0.1, 0.7, 0.2],
        'verified_income': 0.3,
    },
}

SEGMENT_WEIGHTS = [0.35, 0.30, 0.20, 0.15]  # Distribution across segments

def generate_customer_data(n_samples=5000):
    samples_per_segment = np.random.multinomial(n_samples, SEGMENT_WEIGHTS)
    
    records = []
    for seg_id, (seg_name, n) in enumerate(zip(
        ['Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk'],
        samples_per_segment
    )):
        params = SEGMENT_PARAMS[seg_id]
        
        income = np.random.normal(params['income'][0], params['income'][1], n).clip(15000, 300000)
        credit_score = np.random.normal(params['credit_score'][0], params['credit_score'][1], n).clip(300, 850).astype(int)
        employment_years = np.random.exponential(params['employment_years'][0], n).clip(0, 40)
        debt_to_income = np.random.normal(params['debt_to_income'][0], params['debt_to_income'][1], n).clip(0.01, 0.8)
        loan_history_count = np.random.poisson(params['loan_history_count'][0], n).clip(0, 20)
        age = np.random.normal(params['age'][0], params['age'][1], n).clip(18, 80).astype(int)
        
        home_ownership_probs = params['home_ownership']
        home_ownership_vals = np.random.choice([1, 2, 3], n, p=home_ownership_probs)
        
        verified_income = np.random.binomial(1, params['verified_income'], n)
        
        for i in range(n):
            records.append({
                'income': round(income[i], 2),
                'credit_score': credit_score[i],
                'employment_years': round(employment_years[i], 2),
                'debt_to_income': round(debt_to_income[i], 4),
                'loan_history_count': loan_history_count[i],
                'age': age[i],
                'home_ownership': home_ownership_vals[i],
                'verified_income': verified_income[i],
                'segment_label': seg_id,
                'segment_name': seg_name,
            })
    
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_customer_data(5000)
    print(df.head())
    print(f"\nSegment distribution:\n{df['segment_name'].value_counts()}")
    df.to_csv('customers_raw.csv', index=False)
    print("\nSaved to customers_raw.csv")