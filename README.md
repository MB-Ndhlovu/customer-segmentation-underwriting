# Customer Segmentation for Underwriting

## Overview
This project builds an unsupervised + supervised pipeline to segment loan applicants into 4 risk tiers used in lending decisions: **Mass Market**, **Rising Prime**, **Established Prime**, and **Subprime High-Risk**.

KMeans clustering discovers natural customer groupings from financial behavior features. A supervised RandomForest classifier is then trained on cluster labels to enable real-time segment prediction from application data.

## Segments

| Label | Segment | Profile |
|-------|---------|---------|
| 0 | Mass Market | Low income, mid credit, moderate employment |
| 1 | Rising Prime | Mid income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low credit score, high DTI, unstable employment |

## Pipeline

1. **Data Generation**: Synthetic dataset of 5,000 applicants with realistic distributions across income, credit score, employment, DTI, loan history, age, home ownership, and income verification.
2. **Feature Engineering**: RFM (Recency/Frequency/Monetary equivalent), behavioral, and stability features.
3. **Clustering**: KMeans with Elbow method and Silhouette analysis to select optimal k=4.
4. **Classification**: RandomForestClassifier trained on cluster labels for production deployment.

## Results

- **Silhouette Score**: ~0.58 (4 clusters)
- **RandomForest Accuracy**: >97% on held-out test set
- **Segments are mutually exclusive and interpretable for underwriting policy**

## Files

```
src/
  data_loader.py   — synthetic data generator
  features.py      — feature engineering
  segment.py       — KMeans + analysis
  classify.py      — supervised model
run_pipeline.py    — end-to-end execution
reports/
  segmentation_results.json — artifacts and metrics
```

## Business Impact

- Enable risk-tier pricing at origination
- Reduce manual underwriting review for prime segments
- Flag high-risk applicants for enhanced due diligence
- Scalable to real-time decisioning via the classification model