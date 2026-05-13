# Customer Segmentation for Underwriting

## Overview
Machine learning pipeline for customer segmentation in lending/underwriting contexts. Uses KMeans clustering to identify 4 distinct borrower segments, then trains a supervised classifier to predict segment membership from application features alone.

## Segments
| Label | Segment | Risk Profile |
|-------|---------|--------------|
| 0 | Mass Market | Standard lending candidates |
| 1 | Rising Prime | Growing creditworthiness |
| 2 | Established Prime | Low-risk, high-stability borrowers |
| 3 | Subprime High-Risk | Elevated default risk — requires scrutiny |

## Methodology
1. **Data Generation**: Synthetic dataset of 5,000 customers with realistic distributions across income, credit score, employment history, DTI, loan history, age, home ownership, and income verification status.
2. **Feature Engineering**: RFM features (recency, frequency, monetary), behavioral proxies, and stability indicators.
3. **Clustering**: KMeans with Elbow method and Silhouette analysis to determine k=4.
4. **Classification**: RandomForest trained on cluster labels to enable segment prediction for new applicants.

## Results
- Optimal clusters: 4 (validated via silhouette score and elbow method)
- Supervised model accuracy on held-out test set: ~94%
- Segment distributions align with lending industry intuition

## Files
- `src/data_loader.py` — Synthetic data generation (5,000 rows)
- `src/features.py` — Feature engineering (RFM, behavioral, stability)
- `src/segment.py` — KMeans clustering + validation
- `src/classify.py` — RandomForest classifier on cluster labels
- `run_pipeline.py` — End-to-end pipeline orchestration

## Business Impact
- Enables risk-tiered underwriting decisions
- Supports pricing strategy by segment
- Identifies which applicants need manual review (Segment 3)