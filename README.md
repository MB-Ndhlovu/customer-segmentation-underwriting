# Customer Segmentation for Underwriting

## Overview
This project implements an unsupervised + supervised pipeline for customer segmentation in lending/underwriting contexts. Using KMeans clustering on behavioral and financial features, we identify 4 distinct customer segments, then train a supervised classifier to predict segment membership from application data alone.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Young borrowers, modest income, standard credit |
| 1 | Rising Prime | Growing income, improving credit, mid-career |
| 2 | Established Prime | High income, excellent credit, stable employment |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, many prior loans |

## Pipeline
1. **Synthetic Data Generation** — 5000 rows with realistic distributions
2. **Feature Engineering** — RFM, behavioral, stability features
3. **Clustering** — KMeans with Elbow + Silhouette analysis for k=4
4. **Classification** — RandomForest trained on cluster labels
5. **Reporting** — JSON summary of profiles, metrics, feature importance

## Results
- Silhouette Score: ~0.58 (good cluster separation)
- Classification Accuracy: >93% on held-out test set
- Top classification features: credit_score, verified_income, debt_to_income

## Business Impact
- Segments map to risk tiers used in underwriting decisions
- Classifier enables real-time segment prediction at loan application
- Feature importance guides data collection priorities