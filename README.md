# Customer Segmentation for Underwriting

## Overview
This project segments loan applicants into 4 distinct risk tiers using unsupervised clustering (KMeans) on engineered financial features. A supervised classifier is then trained to predict segment membership from application data — enabling fast, explainable underwriting decisions.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, strong credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, short employment |

## Pipeline
1. **data_loader** — generates 5,000 synthetic customer records
2. **features** — builds RFM, behavioral, and stability features
3. **segment** — KMeans clustering with Elbow + Silhouette analysis
4. **classify** — RandomForest classifier trained on cluster labels

## Results
- Silhouette score and cluster profiles saved to `reports/segmentation_results.json`
- Supervised model enables real-time segment prediction for new applicants

## Business Impact
- Risk-stratified lending decisions
- Reduced default rates via targeted product matching
- Explainable segment labels for compliance