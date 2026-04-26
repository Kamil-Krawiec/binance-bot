# Module 1 Plan: Clean Foundation And Master Dataset Builder

## Summary
- Save this plan in-repo for future reference.
- Archive research-only clutter into `ARCHIVE/`, keeping current `stage1/` and `spot/` code available until replacements exist.
- Build a clean, scalable package for the first production dataset path: master dataset for `backW=5`, `forW=2`.
- Reuse current notebook feature creation, but move it into tested modules.
- Treat additional labels as additional features/descriptors, not model targets.
- Paper reference: local `Crypto bot.pdf`; verified against the open-access paper at <https://iris.polito.it/retrieve/handle/11583/2983027/682057>.

## Module 1 Changes
- Add package structure under `src/binance_bot/`.
- Add `pyproject.toml` so the package, CLI, and tests are standard and reproducible.
- Add `configs/master_dataset.yaml` with the current default artifact paths and `backW=5`, `forW=2`.
- Update `.gitignore` so source and planning files can be tracked while large generated data, caches, `.env`, virtualenvs, and test/build artifacts remain ignored.

## Archive Policy
- Create `ARCHIVE/research/`.
- Move exploratory notebooks and notebook-exported story scripts there:
  - `notebooks/*.ipynb`
  - `notebooks/stage2_hold_extremes.py`
  - `notebooks/stage2_hold_extremes_story.py`
  - `stage1/notebook_stage_2.ipynb`
- Keep these active for now:
  - `stage1/history_loader.py`
  - `stage1/pipeline_stage1.py`
  - `stage1/run_stage1.py`
  - `stage1/indicators.py`
  - `spot/*`
- Do not move generated data in Module 1.

## Master Dataset Features
- Preserve paper target columns: `label`, `label_id`, `backW`, `forW`, `alpha`, `beta`, and `fee`.
- Reuse current feature creation from `stage3_master_dataset.ipynb`:
  - RSI, ULTOSC, pct change
  - Bollinger fields
  - close/volume z-scores
  - EMA crossover features
  - month, day of week, hour
  - candlestick pattern columns when available
- Add optional Fear & Greed features from `cache/fear_greed_daily.csv`:
  - `fng`
  - encoded classification
  - daily change, short rolling change, and simple fear/greed regime flags
- Add an exogenous-feature interface for later S&P 500 / AlphaVantage market context, but do not require network/API fetching in Module 1.
- Keep exogenous features optional: if the CSV is missing, the builder still works and records that the feature group was skipped.

## Functional Test Plan
- Test that a tiny fixture build produces a parquet with required identifiers, targets, and feature columns.
- Test that labels join to OHLCV by timestamp without row duplication.
- Test that the builder processes multiple symbols, preventing the current one-symbol notebook limitation.
- Test that optional Fear & Greed joins by date and does not break when missing.
- Test that output has no feature NaNs after warm-up trimming.
- Avoid tests tied to implementation details like private helper names or exact internal call order.

## Assumptions
- Module 1 is foundation plus master dataset builder only; no model training yet.
- First real output remains `backW=5`, `forW=2`.
- S&P 500 / AlphaVantage features are planned as the next exogenous-feature expansion after local dataset generation is stable.
- Existing generated artifacts stay where they are until a later data-layout phase.
