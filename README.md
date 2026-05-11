# Customer Segmentation for Underwriting

## Overview
This project builds an unsupervised + supervised pipeline for customer segmentation in a lending/underwriting context. Using synthetic financial data for 5,000 customers, KMeans clustering identifies 4 distinct risk tiers, which are then used to train a supervised classifier for real-time segment prediction.

## Methodology

### 1. Data Generation (`src/data_loader.py`)
Synthetic dataset of 5,000 customers with features relevant to lending decisions:
- `income` — annual income in ZAR
- `credit_score` — normalized credit score (300–850)
- `employment_years` — years at current employer
- `debt_to_income` — monthly debt / monthly income ratio
- `loan_history_count` — number of previous loans
- `age` — customer age
- `home_ownership` — categorical: own/mortgage/rent/other
- `verified_income` — boolean

Segments are engineered by sampling parameters per cluster so KMeans recovers meaningful groups.

### 2. Feature Engineering (`src/features.py`)
Three feature families:
- **RFM features** — financial health indicators (debt-to-income, verified income flag)
- **Behavioral features** — loan history, employment tenure
- **Stability features** — age, home ownership encoding

### 3. Clustering & Segmentation (`src/segment.py`)
- StandardScaler normalization
- Elbow method (inertia across k=1–10)
- Silhouette analysis for optimal k
- KMeans with k=4
- Segment profiling: mean/std per cluster across all features

### 4. Supervised Classification (`src/classify.py`)
- RandomForestClassifier trained on cluster labels
- Train/test split (80/20)
- Evaluation: accuracy, classification report, feature importance
- Purpose: predict segment label from application features alone

## Results

### Segment Profiles

| Segment | Label | Income (ZAR) | Credit Score | Emp. Years | DTI | Loan Count | Age | Home Ownership | Verified |
|---------|-------|-------------|--------------|------------|-----|------------|-----|----------------|---------|
| 0 | Mass Market | ~380,000 | ~545 | ~2.5 | ~0.28 | ~2.5 | ~34 | rent (58%) | 45% |
| 1 | Rising Prime | ~620,000 | ~665 | ~4.5 | ~0.20 | ~2.0 | ~40 | mortgage (42%) | 72% |
| 2 | Established Prime | ~1,050,000 | ~765 | ~8.5 | ~0.13 | ~1.2 | ~48 | own (55%) | 95% |
| 3 | Subprime High-Risk | ~210,000 | ~435 | ~1.5 | ~0.48 | ~4.5 | ~29 | rent (72%) | 22% |

### Model Performance
- **Clustering**: k=4 selected via silhouette analysis
- **Classifier**: RandomForest accuracy ~94% on test set
- **Top predictive features**: credit_score, income, debt_to_income, age

## Business Impact
- **Mass Market** (40%): Standard lending products, automated approval flows
- **Rising Prime** (28%): Growth-tier offers, premium rate tiers
- **Established Prime** (20%): Low-risk, premium credit cards, relationship banking
- **Subprime High-Risk** (12%): Manual review required, higher rates or decline pathways

## Files
- `src/data_loader.py` — data generation
- `src/features.py` — feature engineering
- `src/segment.py` — clustering pipeline
- `src/classify.py` — supervised model
- `run_pipeline.py` — orchestrates full pipeline
- `reports/segmentation_results.json` — saved results