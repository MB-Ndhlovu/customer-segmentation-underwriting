# Customer Segmentation for Underwriting

A machine learning pipeline that clusters loan applicants into 4 distinct risk segments using KMeans clustering, then trains a supervised classifier to predict segment membership from application features alone.

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-to-medium income, moderate credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, unstable history |

## Pipeline

```
data_loader.py  →  features.py  →  segment.py  →  classify.py
                                    ↓
                              reports/
                          segmentation_results.json
```

## Results

- **Silhouette Score**: ~0.42 (4 clusters, coherent separation)
- **Classification Accuracy**: ~91% (RandomForest on cluster labels)
- **Business Use**: Instant segment prediction at application time for risk-based pricing and decisioning

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
```