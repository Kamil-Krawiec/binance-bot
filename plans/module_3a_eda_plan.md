# Module 3A Plan: Comprehensive EDA, Feature Diagnostics, And Normalization Readiness

## Summary
- Before model training, add a dedicated EDA module that explains what influences `label_id` / `label`, checks feature quality, and decides which features need scaling or removal.
- Reuse `data-science-analysis-library==1.1` where useful, especially feature distributions, category proportions, outlier detection, and feature importance.
- Add robust summary tables and charts because some library functions are display-oriented and not ideal for repeatable pipeline output.
- Save the plan and all EDA artifacts for later review before implementing the ML training module.

## Key Changes
- Add config and CLI:
  - `configs/eda_analysis.yaml`
  - CLI command: `run-eda-analysis`
- Add an EDA runner that loads:
  - `train_b5_f2.parquet`
  - `validation_b5_f2.parquet`
  - `feature_manifest_b5_f2.json`
  - `xgboost_feature_importance_scores.csv`
- Output everything under:
  - `stage1/stage2_data/eda/`

## EDA Outputs
- Dataset overview: row counts, date ranges, symbol counts, label distribution, missing/inf checks, drift, and per-symbol label distribution.
- Target influence analysis: correlation with `label_id`, mutual information, one-vs-rest correlations, and merged XGBoost importance.
- Feature diagnostics: missing/inf, zero variance, outliers, highly correlated pairs, and train-validation stability.
- Visualizations: target distribution, top influence rankings, distributions by target, drift plots, correlation heatmap, and symbol label distributions.
- Normalization recommendations for tree models, MLP/logistic models, binary/time columns, and heavy-tailed features.

## Assumptions
- `label_id` is the numeric target for analysis, while `label` is used for readable plots.
- EDA samples expensive visualizations but uses full data for practical summary tables.
- EDA does not mutate train/validation data; final feature pruning and normalization are applied in the later ML module.
