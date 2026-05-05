# Customer Segmentation for Underwriting

## Overview
This project implements a customer segmentation system for lending underwriting using unsupervised learning (KMeans clustering) and supervised classification (Random Forest). The system segments loan applicants into 4 actionable risk tiers.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard risk |
| 1 | Rising Prime | Good income & credit, stable employment, low risk |
| 2 | Established Prime | High income, excellent credit, very low risk |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, elevated risk |

## Features Used
- **income**: Annual income
- **credit_score**: Credit score (300-850)
- **employment_years**: Years at current employer
- **debt_to_income**: Monthly debt payments / monthly income
- **loan_history_count**: Number of previous loans
- **age**: Customer age
- **home_ownership**: Own/Rent/Mortgage (encoded)
- **verified_income**: Whether income is verified (bool)

## Pipeline
1. Generate 5000 synthetic customer records
2. Engineer RFM, behavioral, and stability features
3. Cluster with KMeans (k=4), validate via Silhouette + Elbow
4. Profile each segment
5. Train Random Forest classifier on cluster labels
6. Save segmentation results to JSON

## Business Impact
- Enables risk-based pricing for loan products
- Supports automated underwriting decisions
- Identifies high-risk applicants requiring manual review
- Targets marketing to prime customers

## Results Summary
Run `python run_pipeline.py` to execute and view cluster profiles and model metrics.