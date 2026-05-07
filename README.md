# Customer Segmentation for Underwriting

## Overview

This project applies unsupervised learning (KMeans clustering) to segment loan applicants into 4 distinct risk tiers, then trains a supervised classifier to predict segment membership from application features alone — enabling real-time underwriting decisions without retraining the clustering model.

## Segments

| Label | Segment Name | Profile |
|-------|-------------|---------|
| 0 | Mass Market | Low-to-mid income, moderate credit, short-to-medium employment, standard debt load |
| 1 | Rising Prime | Growing income, improving credit, mid-career stability, low leverage |
| 2 | Established Prime | High income, excellent credit, long tenure, low debt, home-owning |
| 3 | Subprime High-Risk | Low income, poor credit, short employment, high debt-to-income, unverified income |

## Architecture

```
data_loader.py  →  features.py  →  segment.py  →  classify.py
     ↓                  ↓                ↓              ↓
  Raw CSV          Feature eng.    KMeans labels    RF classifier
                   (RFM, behav,    Silhouette/Elbow  predict segment
                    stability)     Segment profiles  from application
```

## Results Summary

- **Silhouette Score**: ~0.42 (good cluster separation)
- **Supervised Accuracy**: ~96% (RF predicts cluster from features with high fidelity)
- **Feature Importance**: credit_score, income, debt_to_income are dominant

## Business Impact

- Segment 3 (Subprime High-Risk) → auto-decline or manual review with elevated scrutiny
- Segment 2 (Established Prime) → expedited approval, premium rate tier
- Segment 1 (Rising Prime) → standard approval pipeline
- Segment 0 (Mass Market) → standard review, conservative rate

## Files

- `src/data_loader.py` — Generates 5000-row synthetic customer dataset
- `src/features.py` — RFM, behavioral, stability feature engineering
- `src/segment.py` — KMeans clustering, silhouette/elbow analysis, profiling
- `src/classify.py` — RandomForest classifier trained on cluster labels
- `run_pipeline.py` — End-to-end execution
- `reports/segmentation_results.json` — Artifacts and metrics