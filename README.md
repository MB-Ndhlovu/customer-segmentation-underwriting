# Customer Segmentation for Underwriting

Machine learning pipeline for customer segmentation using KMeans clustering and supervised classification for credit risk underwriting.

## Overview

This project builds a 4-segment customer classification system for lenders:
- **Mass Market** — Low-risk, standard products
- **Rising Prime** — Growing credit profiles, good trajectory
- **Established Prime** — Stable, high-credit-quality borrowers
- **Subprime High-Risk** — Elevated risk, requires additional due diligence

## Architecture

```
data_loader.py  →  features.py  →  segment.py  →  classify.py
    ↓                  ↓               ↓              ↓
 5000 rows       RFM features    KMeans clustering  RandomForest
 synthetic data  stability       silhouette eval    classifier
                  behavioral      elbow method       segment prediction
```

## Pipeline

1. **Data Generation**: Synthetic dataset of 5000 customers with realistic distributions
2. **Feature Engineering**: RFM, behavioral, and stability features
3. **Clustering**: KMeans with silhouette analysis and elbow method validation
4. **Classification**: RandomForest trained on cluster labels for production prediction

## Results

Run `python run_pipeline.py` to execute the full pipeline and generate `reports/segmentation_results.json`.

## Business Impact

- Faster applicant segment classification
- Consistent risk-tier assignment across underwriting teams
- Explainable segment predictions via RandomForest feature importance