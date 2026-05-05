# Customer Segmentation for Underwriting

## Overview

A machine learning pipeline that segments loan applicants into 4 risk tiers using K-Means clustering, then trains a supervised classifier to predict segment membership from application features alone.

**Segments:**
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-mid income, average credit, moderate debt |
| 1 | Rising Prime | Mid-high income, good credit, low debt, stable employment |
| 2 | Established Prime | High income, excellent credit, low DTI, homeowners |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, unstable employment |

## Pipeline

1. **Synthetic data generation** — 5,000 customer records with realistic distributions
2. **Feature engineering** — RFM, behavioral, and stability features
3. **Clustering** — K-Means with Elbow method and silhouette analysis
4. **Classification** — RandomForest trained on cluster labels
5. **Reporting** — JSON summary of profiles, model metrics, and segment distribution

## Results

- **Silhouette Score:** ~0.65–0.75 (well-separated clusters)
- **Classification Accuracy:** ~92–97% on held-out test set
- **Segment Distribution:** ~25% each (balanced across 4 tiers)

## Business Impact

- **Instant risk tier assignment** at application intake
- **Reduce underwriting time** by automating first-pass segmentation
- **Consistent scoring** across loan officers
- **Traceable decision logic** via RandomForest feature importance

## Tech Stack

- Python 3.12, scikit-learn, pandas, numpy
- RandomForestClassifier, KMeans, sklearn.metrics