import numpy as np
import pandas as pd

def generate_customers(n=5000, seed=42):
    np.random.seed(seed)
    
    # Segment proportions to ensure 4 distinct groups
    segment_props = [0.35, 0.30, 0.20, 0.15]
    
    customers = []
    
    # Segment 0: Mass Market (low-mid income, average credit, young)
    n0 = int(n * segment_props[0])
    income0 = np.random.normal(38000, 12000, n0).clip(15000, 90000)
    credit0 = np.random.normal(640, 60, n0).clip(520, 740)
    emp0 = np.random.exponential(2.5, n0).clip(0, 15)
    dti0 = np.random.normal(0.32, 0.10, n0).clip(0.05, 0.60)
    loan0 = np.random.poisson(1.5, n0).clip(0, 8)
    age0 = np.random.normal(28, 5, n0).clip(18, 45)
    home0 = np.random.choice(['rent', 'rent', 'rent'], n0)
    verify0 = np.random.choice([True, False], n0, p=[0.4, 0.6])
    customers.extend([
        income0, credit0, emp0, dti0, loan0, age0, home0, verify0
    ])
    
    # Segment 1: Rising Prime (moderate income, improving credit, stable employment)
    n1 = int(n * segment_props[1])
    income1 = np.random.normal(65000, 18000, n1).clip(35000, 120000)
    credit1 = np.random.normal(710, 50, n1).clip(620, 800)
    emp1 = np.random.exponential(5.0, n1).clip(1, 20)
    dti1 = np.random.normal(0.28, 0.08, n1).clip(0.10, 0.50)
    loan1 = np.random.poisson(2.5, n1).clip(0, 10)
    age1 = np.random.normal(35, 6, n1).clip(22, 55)
    home1 = np.random.choice(['rent', 'own', 'own'], n1)
    verify1 = np.random.choice([True, False], n1, p=[0.65, 0.35])
    customers.extend([income1, credit1, emp1, dti1, loan1, age1, home1, verify1])
    
    # Segment 2: Established Prime (high income, excellent credit, long tenure)
    n2 = int(n * segment_props[2])
    income2 = np.random.normal(110000, 35000, n2).clip(70000, 250000)
    credit2 = np.random.normal(770, 40, n2).clip(700, 850)
    emp2 = np.random.exponential(10.0, n2).clip(3, 35)
    dti2 = np.random.normal(0.22, 0.06, n2).clip(0.05, 0.40)
    loan2 = np.random.poisson(3.5, n2).clip(0, 15)
    age2 = np.random.normal(45, 7, n2).clip(30, 65)
    home2 = np.random.choice(['own', 'own', 'own', 'mortgage'], n2)
    verify2 = np.random.choice([True, False], n2, p=[0.85, 0.15])
    customers.extend([income2, credit2, emp2, dti2, loan2, age2, home2, verify2])
    
    # Segment 3: Subprime High-Risk (low income, poor credit, high DTI, verified income issues)
    n3 = n - n0 - n1 - n2
    income3 = np.random.normal(28000, 10000, n3).clip(10000, 60000)
    credit3 = np.random.normal(560, 55, n3).clip(450, 620)
    emp3 = np.random.exponential(1.8, n3).clip(0, 10)
    dti3 = np.random.normal(0.45, 0.12, n3).clip(0.20, 0.75)
    loan3 = np.random.poisson(3.0, n3).clip(0, 12)
    age3 = np.random.normal(32, 7, n3).clip(18, 55)
    home3 = np.random.choice(['rent', 'rent', 'lease'], n3)
    verify3 = np.random.choice([True, False], n3, p=[0.25, 0.75])
    customers.extend([income3, credit3, emp3, dti3, loan3, age3, home3, verify3])
    
    df = pd.DataFrame({
        'income': np.concatenate([income0, income1, income2, income3]),
        'credit_score': np.concatenate([credit0, credit1, credit2, credit3]),
        'employment_years': np.concatenate([emp0, emp1, emp2, emp3]),
        'debt_to_income': np.concatenate([dti0, dti1, dti2, dti3]),
        'loan_history_count': np.concatenate([loan0, loan1, loan2, loan3]),
        'age': np.concatenate([age0, age1, age2, age3]),
        'home_ownership': np.concatenate([home0, home1, home2, home3]),
        'verified_income': np.concatenate([verify0, verify1, verify2, verify3]),
    })
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['home_ownership'] = df['home_ownership'].astype(str)
    df['verified_income'] = df['verified_income'].astype(bool)
    
    return df

if __name__ == "__main__":
    df = generate_customers()
    print(df.describe())
    print(df['home_ownership'].value_counts())