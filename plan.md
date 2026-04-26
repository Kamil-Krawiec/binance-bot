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


# Module 3 Plan: Baseline ML Models, Optuna Tuning, Paper-Faithful Backtest

## Summary
- Confirmed from the paper: `SELL` is a **long-position close signal**, not a short entry. The paper’s backtest enters on first `BUY`, holds through `BUY/HOLD`, closes on next `SELL`, and writes the strategy as `(Buy[Buy|Hold]*Sell)+`; it also describes the comparison strategy as buy-only. Source: Politecnico PDF, Backtesting section: https://iris.polito.it/retrieve/handle/11583/2983027/682057
- Implement Module 3 as **paper long-only first**: no futures shorts yet.
- Train and compare `DummyClassifier`, `RandomForestClassifier`, `MLPClassifier`, and `XGBClassifier`.
- Use Optuna to tune real trading behavior: validation backtest return/profit factor with guardrails for minimum trades and max drawdown.
- Keep validation unbalanced and untouched; use the current balanced train parquet only for fitting.

## Key Changes
- Add model config and CLI:
  - `configs/model_baseline.yaml`
  - CLI command: `train-baseline-models`
- Add model pipeline modules:
  - training loads `train_b5_f2.parquet`, `validation_b5_f2.parquet`, and `feature_manifest_b5_f2.json`
  - feature list comes only from the manifest
  - raw execution prices are joined from `master_dataset_b5_f2.parquet` by `symbol, ts`
- Models:
  - Dummy baseline using training label prior
  - Random Forest with class weights
  - sklearn MLP with `StandardScaler`
  - XGBoost multiclass classifier
- Optuna:
  - install/add `optuna`
  - tune XGBoost, Random Forest, and MLP with small bounded search spaces
  - tune trading thresholds:
    - `buy_threshold`
    - `sell_threshold`
    - `stop_loss`
  - default stop-loss grid/range should include paper values: `0`, `0.01`, `0.025`, `0.05`, `0.10`
- Backtest:
  - state machine: `FLAT` or `LONG`
  - if `FLAT` and `P(BUY) >= buy_threshold`, enter long at signal candle open
  - if `LONG` and `P(SELL) >= sell_threshold`, exit at signal candle open
  - repeated `BUY` while long does not add another position
  - `SELL` while flat does nothing
  - close remaining open positions at final available close
  - include 0.1% fee per entry and exit
  - apply stop-loss while in position using candle low/open-close data from master
- Outputs under `stage1/stage2_data/models/`:
  - fitted model files
  - `model_comparison.json`
  - `threshold_search.csv`
  - `optuna_trials.csv`
  - validation predictions parquet
  - charts: confusion matrix, equity curve, drawdown curve, trades by symbol, threshold heatmap

## Training And Evaluation
- ML metrics:
  - macro F1
  - balanced accuracy
  - per-class precision/recall for `BUY`, `HOLD`, `SELL`
  - confusion matrix
- Trading metrics:
  - total ROI
  - profit factor
  - max drawdown
  - number of trades
  - win rate
  - average win/loss
  - average holding candles
  - trades per symbol
- Optuna objective:
  - maximize validation ROI/profit factor
  - reject/penalize trials with too few trades, excessive drawdown, or invalid probability behavior
  - keep full trial history for review, not only the winner

## Test Plan
- Functional test that each model type can train on a tiny fixture and emit class probabilities.
- Backtest tests:
  - `BUY ... SELL` opens and closes one long trade
  - repeated `BUY` while long does not pyramid
  - `SELL` while flat does not short
  - open position is closed at final candle
  - stop-loss exits as expected
  - fees reduce PnL
- Optuna smoke test with 1-2 trials to verify objective wiring without expensive training.
- CLI smoke test using tiny fixture paths.

## Assumptions
- First production backtest is long-only because that matches the paper.
- Short/futures logic is deferred until after long-only model quality is understood.
- MLP uses sklearn first for speed and simplicity; PyTorch can come later if sklearn MLP is promising.
- Model selection prioritizes validation trading performance, not raw accuracy.
