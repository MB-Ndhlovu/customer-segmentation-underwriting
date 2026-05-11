# Customer Segmentation for Underwriting

## Overview
Unsupervised KMeans clustering combined with supervised RandomForest classification to segment customers for life insurance underwriting. Produces 4 actionable segments from application features.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Young, low credit, moderate debt, entry-level |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure, low risk |
| 3 | Subprime High-Risk | Low credit, high DTI, thin file, elevated risk |

## Pipeline
1. `src/data_loader.py` — Generate 5000 synthetic customer records
2. `src/features.py` — Engineer RFM, behavioral, stability features
3. `src/segment.py` — KMeans + silhouette/elbow analysis → cluster labels
4. `src/classify.py` — RandomForest trained on cluster labels
5. `run_pipeline.py` — Execute full pipeline, print summary, save `reports/segmentation_results.json`

## Results
- Silhouette score, inertia, cluster profiles saved to `reports/segmentation_results.json`
- RandomForest accuracy on cluster labels included in output