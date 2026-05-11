# Customer Segmentation for Underwriting

## Overview
Unsupervised KMeans clustering + supervised RandomForest classification pipeline for credit underwriting segmentation. Synthetic dataset of 5,000 customer records with 8 features.

## Segments
| Label | Name | Description |
|-------|------|-------------|
| 0 | Mass Market | Low-to-medium income, average credit, stable employment |
| 1 | Rising Prime | Growing income, improving credit, early career |
| 2 | Established Prime | High income, strong credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high debt-to-income |

## Pipeline
```
data_loader.py  →  features.py  →  segment.py  →  classify.py
      ↓                ↓               ↓              ↓
  raw customer    engineered      KMeans +       RandomForest
  records (5k)    features        profiling      classifier
```

## Files
- `src/data_loader.py` — synthetic data generation
- `src/features.py` — RFM, behavioral, stability features
- `src/segment.py` — KMeans clustering, silhouette, elbow, profiling
- `src/classify.py` — RandomForest trained on cluster labels
- `run_pipeline.py` — full pipeline execution

## Results
Saved to `reports/segmentation_results.json`