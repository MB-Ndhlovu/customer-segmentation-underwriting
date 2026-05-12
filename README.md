# Customer Segmentation for Underwriting

## Overview
This project implements an unsupervised customer segmentation system for lending underwriting using KMeans clustering. The model identifies 4 distinct customer segments that lenders use to assess risk and tailor loan products.

## Segments
| Segment | Profile | Risk Level |
|---------|---------|------------|
| 0 | Mass Market | Medium |
| 1 | Rising Prime | Low |
| 2 | Established Prime | Low |
| 3 | Subprime High-Risk | High |

## Pipeline
1. `src/data_loader.py` — Generate 5000 synthetic customer records
2. `src/features.py` — Engineer RFM, behavioral, and stability features
3. `src/segment.py` — KMeans clustering with Elbow + Silhouette analysis
4. `src/classify.py` — RandomForest classifier trained on cluster labels
5. `run_pipeline.py` — End-to-end execution with reporting

## Results
- **Silhouette Score**: captures cluster cohesion and separation
- **Segment Profiles**: centroid statistics per cluster
- **Classifier Accuracy**: supervised model trained on cluster assignments
- **Feature Importance**: which input features drive segment assignment

## Business Impact
- Enables risk-based pricing for loan products
- Supports credit policy decisions with data-driven segments
- Provides interpretable customer buckets for underwriting teams