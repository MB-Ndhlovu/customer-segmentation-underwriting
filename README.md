# Customer Segmentation for Underwriting

A machine learning pipeline that segments potential borrowers into four risk tiers using unsupervised clustering (KMeans), then deploys a supervised classifier to predict segment membership from application features alone.

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-to-mid income, moderate credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, strong credit, long tenure, low DTI |
| 3 | Subprime High-Risk | Low income, poor credit, short tenure, high DTI |

## Pipeline

```
run_pipeline.py
├── data_loader   → 5000 synthetic customer records
├── features     → RFM, behavioural, stability features
├── segment      → KMeans (k=4), silhouette analysis, elbow method, profiling
└── classify     → RandomForestClassifier on cluster labels
```

## Business Impact

- Risk-adjusted pricing by segment
- Early warning for high-risk applications
- Tailored credit terms per tier
- Reduced default rates through targeted underwriting

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```