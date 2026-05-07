# Customer Segmentation for Underwriting

## Overview
This project implements unsupervised customer segmentation using KMeans clustering, followed by a supervised classification model to predict segment membership from application features. Designed for lending/underwriting workflows.

## Business Context
Four actionable segments for risk-based underwriting:
- **Mass Market** (Label 0): Standard applicants, moderate risk
- **Rising Prime** (Label 1): Young, good credit trajectory
- **Established Prime** (Label 2): Stable, low-risk, verified income
- **Subprime High-Risk** (Label 3): Elevated risk requiring scrutiny

## Methodology

### 1. Data Generation
Synthetic dataset of 5,000 customers with features:
- `income`, `credit_score`, `employment_years`, `debt_to_income`
- `loan_history_count`, `age`, `home_ownership_status`, `verified_income`

### 2. Feature Engineering
- **RFM features**: Derived from recency/frequency/monetary signals
- **Behavioral features**: Debt-to-income patterns, loan frequency
- **Stability features**: Employment tenure, income verification

### 3. Clustering
KMeans with k=4, validated via:
- Elbow method (inertia)
- Silhouette analysis

### 4. Classification
RandomForestClassifier trained on cluster labels — enables real-time segment prediction for new applicants.

## Results
Pipeline produces:
- `reports/segmentation_results.json` — segment profiles and model metrics
- Segment centroids, silhouette score, classification accuracy

## Files
```
src/
  data_loader.py   — synthetic data generation
  features.py      — RFM, behavioral, stability feature engineering
  segment.py       — KMeans clustering + validation
  classify.py      — RandomForest classifier on cluster labels
run_pipeline.py    — end-to-end execution
reports/
  segmentation_results.json — artifacts
```

## Dependencies
scikit-learn, pandas, numpy, json