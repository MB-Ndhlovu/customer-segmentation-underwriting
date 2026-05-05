# Customer Segmentation for Underwriting

## Overview

This project builds a machine learning pipeline that segments loan applicants into four risk categories, enabling lenders to make data-driven credit decisions.

## Segments

| Segment | Label | Profile |
|---------|-------|---------|
| Mass Market | 0 | Low-mid income, average credit, moderate debt |
| Rising Prime | 1 | Mid-high income, good credit, low debt, stable employment |
| Established Prime | 2 | High income, excellent credit, low debt-to-income, homeowners |
| Subprime High-Risk | 3 | Low income, poor credit, high debt, multiple prior loans |

## Pipeline

1. **Data Generation** — 5,000 synthetic customer records with realistic underwriting features
2. **Feature Engineering** — RFM, behavioral, and stability features
3. **Clustering** — KMeans with elbow method and silhouette analysis to determine optimal k=4
4. **Classification** — RandomForest model trained to predict segment from application features
5. **Reporting** — JSON summary of segment profiles and model performance

## Results

- Silhouette Score: ~0.65
- Classification Accuracy: ~94%
- Segments are clearly distinguishable across income, credit score, and debt-to-income

## Business Impact

- Identify high-risk applicants early in the underwriting process
- Price products according to segment risk profile
- Reduce default rates by matching loan terms to borrower capacity

## Files

- `src/data_loader.py` — Synthetic data generation
- `src/features.py` — Feature engineering
- `src/segment.py` — KMeans clustering + profiling
- `src/classify.py` — RandomForest classifier
- `run_pipeline.py` — End-to-end pipeline execution
- `reports/segmentation_results.json` — Output artifacts