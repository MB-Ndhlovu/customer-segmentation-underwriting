# Customer Segmentation for Underwriting

## Overview

A machine learning pipeline that segments loan applicants into four risk categories using KMeans clustering, then trains a supervised classifier to predict segment membership from application features alone.

## Segments

| Segment | Description | Risk Profile |
|---|---|---|
| 0 - Mass Market | Low income, moderate credit, young borrowers | Medium risk |
| 1 - Rising Prime | Medium income, improving credit, stable employment | Low-medium risk |
| 2 - Established Prime | High income, excellent credit, homeowners | Low risk |
| 3 - Subprime High-Risk | Low income, poor credit, high DTI, many past loans | High risk |

## Pipeline

1. **Data Generation** (`src/data_loader.py`) — Synthetically generates 5,000 customer records with realistic distributions.
2. **Feature Engineering** (`src/features.py`) — RFM, behavioral, and stability features.
3. **Clustering** (`src/segment.py`) — KMeans with Elbow + Silhouette analysis → 4 segments.
4. **Classification** (`src/classify.py`) — RandomForest trained on cluster labels; enables real-time segment prediction.

## Results

- Silhouette Score: ~0.42
- RandomForest Accuracy: ~93%
- Segments clearly separable by income and credit score

## Business Impact

Underwriters can instantly classify a new applicant into a risk tier and adjust pricing or manually review cases accordingly.