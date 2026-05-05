# Customer Segmentation for Underwriting

## Overview
This project builds a customer segmentation system for lending underwriting using KMeans clustering and supervised classification. The pipeline segments loan applicants into 4 actionable risk tiers.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Standard applicants, moderate risk |
| 1 | Rising Prime | Young, good income trajectory, low risk |
| 2 | Established Prime | Stable, high credit, low risk |
| 3 | Subprime High-Risk | High DTI, poor credit, elevated default risk |

## Methodology
1. **Data Generation** — Synthetic dataset of 5,000 applicants with realistic feature distributions
2. **Feature Engineering** — RFM, behavioral, and stability features
3. **Clustering** — KMeans (k=4) with Elbow method and Silhouette analysis
4. **Classification** — RandomForest trained on cluster labels for real-time inference

## Results
- Silhouette Score: ~0.42
- Segment distribution roughly: 25% each (balanced)
- RandomForest accuracy on held-out test: ~95%

## Business Impact
- Enable tiered underwriting with differentiated approval criteria
- Reduce default rates by routing high-risk applicants to enhanced review
- Improve portfolio risk monitoring with segment-level tracking

## Files
- `src/data_loader.py` — Synthetic data generation
- `src/features.py` — Feature engineering
- `src/segment.py` — KMeans clustering + profiling
- `src/classify.py` — RandomForest classifier
- `run_pipeline.py` — Full pipeline execution
- `reports/segmentation_results.json` — Output artifacts