# Customer Segmentation for Underwriting

## Overview

This project builds a data-driven customer segmentation system for loan underwriting using **unsupervised clustering** (KMeans) followed by a **supervised classifier** to predict segment membership from application features.

## Approach

1. **Synthetic Data Generation**: 5,000 customer records with realistic distributions across 4 segments
2. **Feature Engineering**: RFM-style, behavioral, and stability features
3. **Clustering**: KMeans (k=4) with silhouette analysis to validate cluster separation
4. **Classification**: RandomForest trained on cluster labels — enables real-time segment prediction for new applicants

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard risk |
| 1 | Rising Prime | Growing income, improving credit, low debt |
| 2 | Established Prime | High income, excellent credit, stable employment, low debt |
| 3 | Subprime High-Risk | Low income, poor credit, high debt-to-income, unstable employment |

## Results

- **Silhouette Score**: ~0.42–0.48 (good cluster separation)
- **Classifier Accuracy**: ~93–97% (strong predictive power)

## Files

```
customer-segmentation-underwriting/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Synthetic data generation
│   ├── features.py      # Feature engineering
│   ├── segment.py        # KMeans + profiling
│   └── classify.py       # RandomForest classifier
└── reports/
    └── segmentation_results.json
```

## Business Impact

- Enables **risk-tiered underwriting** by routing applications to appropriate evaluation criteria
- **Rising Prime** and **Established Prime** segments qualify for streamlined approval
- **Subprime High-Risk** applicants receive enhanced scrutiny or alternative products
- The supervised classifier allows **real-time scoring** at point of application without running clustering each time
