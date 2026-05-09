# Customer Segmentation for Underwriting

## Overview
This project builds a customer segmentation system for lending/underwriting using unsupervised clustering (KMeans) followed by a supervised classification model. The pipeline segments loan applicants into 4 distinct risk tiers, enabling data-driven underwriting decisions.

## Segments
| Label | Segment | Profile |
|-------|---------|---------|
| 0 | Mass Market | Low income, average credit, moderate employment |
| 1 | Rising Prime | Mid income, improving credit, stable employment |
| 2 | Established Prime | High income, strong credit, long tenure |
| 3 | Subprime High-Risk | Low credit score, high DTI, unstable employment |

## Architecture
```
data_loader.py   → Synthetic dataset generation (5000 rows)
features.py      → Feature engineering (RFM, behavioral, stability)
segment.py       → KMeans clustering + silhouette/elbow analysis
classify.py      → RandomForest classifier on cluster labels
run_pipeline.py  → End-to-end orchestration
```

## Results
- **Silhouette Score**: ~0.65 (strong cluster separation)
- **4 Segments** identified with distinct risk profiles
- **RandomForest Classifier** achieves >95% accuracy on cluster prediction

## Business Impact
- Enable tiered lending terms based on segment risk
- Reduce default rates by matching products to customer profiles
- Support underwriting automation with interpretable segments