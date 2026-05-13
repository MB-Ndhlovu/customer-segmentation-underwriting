# Customer Segmentation for Underwriting

## Overview
This project applies unsupervised learning (KMeans clustering) to segment loan applicants into 4 distinct risk tiers, then trains a supervised RandomForest classifier to predict segments from application features alone.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-to-mid income, average credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, unstable employment, high DTI |

## Pipeline
1. **data_loader** — Generates 5000 synthetic customers with realistic distributions
2. **features** — Engineers RFM, behavioral, and stability features
3. **segment** — KMeans clustering with Elbow + Silhouette analysis → 4 clusters
4. **classify** — RandomForest trained on cluster labels for segment prediction

## Results
- **Silhouette Score**: ~0.42 (good separation)
- **Classifier Accuracy**: ~94% on held-out test set
- **Feature Importance**: credit_score, income, employment_years dominate

## Business Impact
- Enable tiered underwriting policies per segment
- Reduce default rates by routing high-risk applicants to enhanced review
- Speed up decisions for prime applicants

## Files
```
├── README.md
├── requirements.txt
├── run_pipeline.py
├── reports/
│   └── segmentation_results.json
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── features.py
    ├── segment.py
    └── classify.py
```
