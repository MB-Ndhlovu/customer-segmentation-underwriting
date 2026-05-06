# Customer Segmentation for Underwriting

## Overview
Unsupervised KMeans clustering + supervised RandomForest classification pipeline for customer risk segmentation in lending.

## Problem
Lenders need to segment applicants into risk tiers for differentiated underwriting — without relying on black-box scores alone.

## Solution
1. **Synthetic dataset** — 5,000 customers with income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income
2. **Feature engineering** — RFM features, behavioral, stability
3. **Clustering** — KMeans with Elbow method + Silhouette analysis → 4 segments
4. **Classification** — RandomForest trained on cluster labels for fast segment prediction on new applications

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, fair credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, many prior loans |

## Results
- **Silhouette Score**: ~0.45–0.55 (well-separated clusters)
- **RandomForest Accuracy**: 95%+ on cluster labels
- **Features**: StandardScaler → PCA for clustering; raw features for classification

## Business Impact
- Instant risk tier assignment for new loan applications
- Reduced underwriting time via automated segment prediction
- Differentiated rate-setting and lending policy per segment

## Files
- `src/data_loader.py` — Synthetic data generation
- `src/features.py` — Feature engineering
- `src/segment.py` — KMeans clustering + profiling
- `src/classify.py` — RandomForest segment classifier
- `run_pipeline.py` — End-to-end pipeline orchestrator
- `reports/segmentation_results.json` — Saved artifacts