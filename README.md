# Customer Segmentation for Underwriting

## Overview

This project implements an unsupervised customer segmentation pipeline for lending underwriting using KMeans clustering, followed by a supervised classification model to predict customer segments from application features.

## Business Problem

Lenders need to categorize loan applicants into meaningful risk tiers for pricing and decisioning. This pipeline identifies 4 distinct customer segments that reflect real-world lending profiles.

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard employment |
| 1 | Rising Prime | Good income growth, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, unstable employment, high DTI |

## Pipeline

1. **Data Generation**: Synthetic dataset of 5000 customers with realistic underwriting features
2. **Feature Engineering**: RFM, behavioral, and stability features
3. **Clustering**: KMeans with elbow method and silhouette analysis
4. **Classification**: RandomForest trained on cluster labels for segment prediction

## Files

```
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Synthetic data generation
│   ├── features.py       # Feature engineering
│   ├── segment.py        # KMeans clustering
│   └── classify.py       # RandomForest classifier
└── reports/
    └── segmentation_results.json
```

## Results

Cluster quality metrics and segment profiles are saved to `reports/segmentation_results.json`.

## Business Impact

- Enables risk-appropriate pricing per segment
- Supports automated underwriting decisions
- Identifies high-risk applicants for manual review
