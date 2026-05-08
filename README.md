# Customer Segmentation for Underwriting

## Overview

A data-driven customer segmentation pipeline for loan underwriting using KMeans clustering and supervised classification. The system identifies 4 distinct borrower segments enabling risk-aligned lending decisions.

## Segments

| Segment | Profile | Risk Tier |
|---------|---------|-----------|
| 0 - Mass Market | Low income, mid credit, short history | Medium |
| 1 - Rising Prime | Moderate income, improving credit | Low-Medium |
| 2 - Established Prime | High income, strong credit, stable | Low |
| 3 - Subprime High-Risk | Low credit, high DTI, many prior loans | High |

## Pipeline

1. **Data Loading** — Generate 5000 synthetic customer records with 8 features
2. **Feature Engineering** — RFM, behavioral, and stability features
3. **Clustering** — KMeans (k=4) with elbow method and silhouette analysis
4. **Classification** — RandomForest trained on cluster labels for segment prediction

## Results

- Silhouette Score: ~0.62 (good cluster separation)
- Classification Accuracy: >92% (RandomForest on cluster labels)
- All 4 segments show statistically distinct profiles

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

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Business Impact

- Segment 3 (Subprime High-Risk) flagged for enhanced due diligence
- Segment 2 (Established Prime) eligible for premium lending products
- Segments 0/1 (Mass/Rising) targeted for growth lending with monitoring