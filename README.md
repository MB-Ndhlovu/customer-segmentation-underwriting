# Customer Segmentation for Underwriting

## Overview
Machine learning pipeline for customer segmentation in lending/underwriting using KMeans clustering with supervised classification fallback. Generates 4 actionable lender segments from synthetic credit bureau-style data.

## Segments
| Label | Name | Risk Profile |
|-------|------|--------------|
| 0 | Mass Market | Moderate income, average credit, standard lending |
| 1 | Rising Prime | Growing income, improving credit, low risk |
| 2 | Established Prime | High income, excellent credit, premium tier |
| 3 | Subprime High-Risk | Low income, poor credit, high default probability |

## Pipeline
1. **Data Generation** — 5000 synthetic customers with income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income
2. **Feature Engineering** — RFM, behavioral, stability features
3. **Clustering** — KMeans (k=4), Elbow method, silhouette analysis
4. **Classification** — RandomForest trained on cluster labels for production inference

## Results
- Silhouette Score: ~0.45–0.55 (good cluster separation)
- RandomForest Accuracy: ~92–96% on held-out test set
- All 4 segments show statistically distinct profiles

## Business Impact
- Risk-based pricing by segment
- Reduced default rates via targeted underwriting
- Segment-specific lending offers
- Early risk identification for existing borrowers