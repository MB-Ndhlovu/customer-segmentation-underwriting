# Customer Segmentation for Underwriting

ML pipeline that clusters 5,000 synthetic credit customers into 4 risk tiers and trains a supervised classifier to predict segment from application features.

## Segments

| ID | Label | Profile |
|----|-------|---------|
| 0 | Mass Market | Young, low income, short employment, high DTI, low credit |
| 1 | Rising Prime | Moderate income, improving credit, mid employment |
| 2 | Established Prime | High income, high credit, long employment, home owners |
| 3 | Subprime High-Risk | Low credit score, high DTI, verified income issues |

## Pipeline

1. `src/data_loader.py` — generates 5,000 synthetic customer records
2. `src/features.py` — RFM, behavioral, stability feature engineering
3. `src/segment.py` — KMeans (k=4), silhouette analysis, elbow method, segment profiling
4. `src/classify.py` — RandomForestClassifier trained on cluster labels

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Business Impact

- Underwriters can route applications by predicted segment
- Risk-based pricing becomes data-driven
- Early identification of subprime applicants enables proactive counseling