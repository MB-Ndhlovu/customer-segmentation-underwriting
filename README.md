# Customer Segmentation for Underwriting

**Goal**: Segment loan applicants into 4 actionable credit tiers using unsupervised clustering, then train a supervised classifier to predict segment membership from application features alone.

## Segments

| Label | Name | Description |
|-------|------|-------------|
| 0 | Mass Market | Low-to-medium income, average credit, stable employment |
| 1 | Rising Prime | Growing income, improving credit, short employment history |
| 2 | Established Prime | High income, excellent credit, long tenure, low risk |
| 3 | Subprime High-Risk | Low income, poor credit, unstable employment, high DTI |

## Pipeline

```
data_loader.py  →  features.py  →  segment.py  →  classify.py
     (5000 rows)      (feature eng)   (KMeans)    (RandomForest)
```

## Results Summary

Run `python run_pipeline.py` to execute the full pipeline and generate `reports/segmentation_results.json`.

## Business Impact

- Underwriters can pre-screen applicants before manual review
- Risk-based pricing becomes possible per segment
- Early warning flags for Subprime High-Risk applicants