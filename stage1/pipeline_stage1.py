from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional
import pandas as pd
# from history_loader import HistoryLoader
try:
    from history_loader import HistoryLoader
except ImportError:
    from .history_loader import HistoryLoader

Label = Literal["BUY", "HOLD", "SELL"]
LABEL_VALUES: List[Label] = ["BUY", "HOLD", "SELL"]
CalibrationMode = Literal["exact_paper", "per_forw_percentile"]


@dataclass
class QAReport:
    """Lightweight quality report for input OHLCV time series."""

    gaps_pct: float
    n_dupes: int
    n_nan: int
    status: str
    notes: str = ""


@dataclass
class Stage1Paths:
    """Filesystem layout for Stage 1 artifacts."""
    processed_dir: Path
    datasets_dir: Path
    reports_dir: Path
    def ensure(self) -> None:
        for folder in (self.processed_dir, self.datasets_dir, self.reports_dir):
            folder.mkdir(parents=True, exist_ok=True)
    @staticmethod
    def _safe_interval(interval: str) -> str:
        return interval.replace("/", "-")
    def processed_file(self, symbol: str, interval: str) -> Path:
        safe_interval = self._safe_interval(interval)
        return self.processed_dir / f"{symbol}_{safe_interval}.parquet"
    def dataset_file(self, symbol: str, interval: str, backW: int, forW: int) -> Path:
        safe_interval = self._safe_interval(interval)
        return self.datasets_dir / f"{symbol}_{safe_interval}_backW{backW}_forW{forW}.parquet"
    def report_base(self, symbol: str, interval: str, backW: int, forW: int) -> Path:
        safe_interval = self._safe_interval(interval)
        return self.reports_dir / f"{symbol}_{safe_interval}_backW{backW}_forW{forW}"



@dataclass
class Stage1Config:
    """
    Immutable configuration bundle for Stage 1 labeling pipeline.

    New fields for "by-the-book" calibration:
      - calibration_mode:
          * "exact_paper": alpha/beta computed globally from 1-candle returns,
                           then beta is scaled by (1 + scale_beta_per_step*(forW-1))
          * "per_forw_percentile": legacy/adaptive mode (percentiles per forW)
      - alpha_const, beta_const: optional hard-coded constants (override)
      - single_candle_percentiles: percentiles for deriving (alpha,beta) from
                                   the Open->Close 1-bar distribution (if not using constants)
      - scale_beta_per_step: multiplicative growth of beta per +1 forward bar,
                             default 0.10 (i.e., +10% per step) per the paper
    """
    symbol: str
    interval: str
    fee: float
    backW_values: List[int]
    forW_values: List[int]
    percentiles_by_forW: Dict[int, Dict[str, float]]  # used only in per_forw_percentile
    paths: Stage1Paths
    calibration_mode: CalibrationMode = "exact_paper"
    alpha_const: Optional[float] = None
    beta_const: Optional[float] = None
    single_candle_percentiles: Dict[str, float] = None  # {"alpha_pct": 0.25, "beta_pct": 0.997}
    scale_beta_per_step: float = 0.10
    start_date: str = "1 Jan 2020"  # for history_loader; not used in labeling

def interval_to_minutes(interval: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    if not interval:
        raise ValueError("Interval string is empty")
    value_part, unit = interval[:-1], interval[-1].lower()
    if unit not in units:
        raise ValueError(f"Unsupported interval unit: {interval}")
    try:
        value = int(value_part)
    except ValueError as exc:
        raise ValueError(f"Invalid interval value: {interval}") from exc
    return value * units[unit]


def _write_dataframe(df: pd.DataFrame, target: Path, *, index: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target, index=index)
        return target
    except (ImportError, ValueError):
        fallback = target.with_suffix(".csv")
        df.to_csv(fallback, index=index)
        return fallback


class DataQualityChecker:
    """Compute QA metrics for OHLCV series."""

    def evaluate(self, df: pd.DataFrame, interval_minutes: int) -> QAReport:
        if df.empty:
            return QAReport(gaps_pct=100.0, n_dupes=0, n_nan=0, status="FAIL", notes="Dataframe is empty")

        freq = pd.Timedelta(minutes=interval_minutes)
        idx = df.index
        duplicates = int(idx.duplicated().sum())
        nan_count = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())

        gaps = 0
        notes: List[str] = []
        if len(idx) > 1:
            deltas = idx.to_series().diff().dropna()
            for delta in deltas:
                ratio = delta / freq
                steps = max(int(round(ratio)), 1)
                if steps > 1:
                    gaps += steps - 1
            if gaps > 0:
                gap_locs = deltas[deltas > freq]
                if not gap_locs.empty:
                    first_gap_loc = gap_locs.index[0]
                    notes.append(f"First gap starts at {first_gap_loc.isoformat()}")

        expected = len(idx) + gaps
        gaps_pct = (gaps / expected * 100.0) if expected else 0.0

        status = "OK"
        if gaps_pct > 10.0 or nan_count > 0:
            status = "FAIL"
        elif gaps_pct > 1.0 or duplicates > 0:
            status = "WARN"

        return QAReport(gaps_pct=gaps_pct, n_dupes=duplicates, n_nan=nan_count, status=status, notes="; ".join(notes))


class ForwardReturnCalculator:
    """Compute forward returns for labeling (paper-consistent)."""

    def __init__(self, fee: float) -> None:
        """
        Args:
            fee: Total round-trip fee as a fraction (e.g., 0.002 for 0.2%).
                 This is subtracted once from the percentage price change.
        """
        self.fee = fee

    def compute(self, df: pd.DataFrame, forW: int) -> pd.Series:
        """
        R_{t,k} = (Close_{t+k} - Open_t) / Open_t - fee

        Notes:
          - Enforces forW >= 1; there is no forW=0 labeling in the paper.
          - Uses only Open_t and Close_{t+k} (no leakage).
        """
        if forW < 1:
            raise ValueError("forW must be >= 1")
        if df.empty:
            return pd.Series(dtype="float64")

        future_close = df["close"].shift(-forW)
        # PAPER-FAITHFUL:
        forward = ((1.0 - self.fee) * future_close - (1.0 + self.fee) * df["open"]) / df["open"]
        forward.name = f"forward_{forW}"
        return forward
    
class EMACalculator:
    """
    Compute EMA over the backward window.

    This is optional for Stage 1 labeling (labels come from forward returns),
    but included to be faithful to the paper's pipeline where EMA(backW) is
    computed as part of the procedure.
    """

    def compute(self, df: pd.DataFrame, backW: int) -> pd.Series:
        """
        Args:
            df: OHLCV DataFrame (must include 'close').
            backW: Backward window length >= 1.

        Returns:
            pandas Series with EMA(backW) aligned to df.index.
        """
        if backW < 1:
            raise ValueError("backW must be >= 1")
        ema = df["close"].ewm(span=backW, adjust=False).mean()
        ema.name = f"ema_backW_{backW}"
        return ema

class ThresholdCalibrator:
    """
    Calibrate alpha/beta thresholds for labeling.

    Modes:
      - exact_paper:
          * Compute (alpha, beta) once from the distribution of 1-candle
            Open->Close percentage changes across the whole dataset
            (or take provided constants).
          * For a given forW, return (alpha, beta * (1 + scale * (forW - 1))).
      - per_forw_percentile:
          * Legacy/adaptive: derive (alpha, beta) from |R_{t,forW}| percentiles.
    """

    def __init__(
        self,
        cfg: Stage1Config
    ) -> None:
        self.cfg = cfg
        self._alpha_global: Optional[float] = None
        self._beta_global: Optional[float] = None


    def _single_candle_abs_returns(self, df: pd.DataFrame) -> pd.Series:
        close_fwd = df["close"].shift(-1)
        r1 = ((1.0 - self.cfg.fee) * close_fwd - (1.0 + self.cfg.fee) * df["open"]) / df["open"]
        return r1.abs().dropna()

    def prepare_global_thresholds(self, df: pd.DataFrame) -> None:
        """
        Prepare global alpha/beta for "exact_paper" mode.

        Priority:
          1) If alpha_const and beta_const are provided -> use them directly.
          2) Else derive from single-candle returns using configured percentiles.
        """
        if self.cfg.alpha_const is not None and self.cfg.beta_const is not None:
            self._alpha_global = float(self.cfg.alpha_const)
            self._beta_global = float(self.cfg.beta_const)
            return
        
        if not self.cfg.single_candle_percentiles:
            self.cfg.single_candle_percentiles = {"alpha_pct": 0.05, "beta_pct": 0.997}

        r1_abs = self._single_candle_abs_returns(df)
        alpha_pct = float(self.cfg.single_candle_percentiles["alpha_pct"])
        beta_pct  = float(self.cfg.single_candle_percentiles["beta_pct"])

        self._alpha_global = float(r1_abs.quantile(alpha_pct))
        self._beta_global  = float(r1_abs.quantile(beta_pct))
        if self._beta_global < self._alpha_global:
            self._beta_global = self._alpha_global
        

    def calibrate(self, forward_abs: pd.Series, forW: int) -> Tuple[float, float]:
        """
        Return (alpha, beta) for the given forward horizon.

        In exact_paper mode:
          - ignores `forward_abs` (uses global thresholds prepared beforehand)
          - returns (alpha_global, beta_global * (1 + scale*(forW-1)))

        In per_forw_percentile mode:
          - computes (alpha, beta) from |R_{t,forW}| percentiles as before.
        """

        if self.cfg.calibration_mode == "exact_paper":
            if self._alpha_global is None or self._beta_global is None:
                raise RuntimeError("Call prepare_global_thresholds(df) first.")
            alpha = self._alpha_global
            beta  = self._beta_global * (1.0 + self.cfg.scale_beta_per_step * (forW - 1))
            if beta < alpha:
                beta = alpha
            return float(alpha), float(beta)

        # Legacy/adaptive mode - maybe worth executing and playing
        params = self.cfg.percentiles_by_forW.get(forW)
        if params is None:
            raise KeyError(f"Percentiles not configured for forW={forW} (per_forw_percentile mode).")
        if forward_abs.empty:
            raise ValueError("Forward returns series is empty for calibration.")

        alpha_pct = float(params.get("alpha_pct"))
        beta_pct  = float(params.get("beta_pct"))
        alpha = float(forward_abs.quantile(alpha_pct))
        beta  = float(forward_abs.quantile(beta_pct))
        if beta < alpha:
            beta = alpha
        return alpha, beta


class LabelAssigner:
    """Assign BUY/HOLD/SELL labels from forward returns and thresholds."""

    def assign(self, forward: pd.Series, alpha: float, beta: float) -> pd.Series:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")

        labels = pd.Series("HOLD", index=forward.index, dtype="object")
        valid = forward.notna()
        labels.loc[~valid] = pd.NA
        if valid.any():
            slice_forward = forward[valid]
            buy_mask = (slice_forward > alpha) & (slice_forward < beta)
            sell_mask = (slice_forward < -alpha) & (slice_forward > -beta)
            labels.loc[buy_mask.index[buy_mask]] = "BUY"
            labels.loc[sell_mask.index[sell_mask]] = "SELL"
        cat_dtype = pd.CategoricalDtype(categories=LABEL_VALUES)
        return labels.astype(cat_dtype)


class WindowIndexer:
    """Enumerate valid sample indices for given (backW, forW)."""

    def build(self, df: pd.DataFrame, backW: int, forW: int) -> pd.DataFrame:
        """
        Ensures both backward context and forward target exist.
        Returns DataFrame with columns ['t','backW','forW'] indexed by valid timestamps.
        """
        if backW < 1:
            raise ValueError("backW must be >= 1")
        if forW < 1:
            raise ValueError("forW must be >= 1")

        n = len(df)
        if n == 0:
            empty_index = pd.DatetimeIndex([])
            return pd.DataFrame(columns=["t", "backW", "forW"], index=empty_index)

        start = backW - 1
        end = n - forW
        if end <= start:
            empty_index = pd.DatetimeIndex([])
            return pd.DataFrame(columns=["t", "backW", "forW"], index=empty_index)

        idx = df.index[start:end]
        return pd.DataFrame({"t": idx, "backW": backW, "forW": forW}, index=idx)


class ArtifactWriter:
    """Persist labeled datasets and per-window reports."""

    def __init__(self, paths: Stage1Paths, symbol: str, interval: str, fee: float) -> None:
        self.paths = paths
        self.symbol = symbol
        self.interval = interval
        self.fee = fee

    def write(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        backW: int,
        forW: int,
        alpha: float,
        beta: float,
    ) -> None:
        cat_dtype = pd.CategoricalDtype(categories=LABEL_VALUES)
        labels = labels.astype(cat_dtype)
        dataset_df = pd.DataFrame(
            {
                "ts": df.index,
                "label": labels.astype("string"),
                "backW": backW,
                "forW": forW,
                "alpha": alpha,
                "beta": beta,
                "fee": self.fee,
                "ema_backW": df.get(f"ema_backW_{backW}"),
            }
        )

        dataset_target = self.paths.dataset_file(self.symbol, self.interval, backW, forW)
        dataset_path = _write_dataframe(dataset_df, dataset_target, index=False)

        total_len = len(labels)
        valid_total = int(labels.count())
        counts = {label: int((labels == label).sum()) for label in LABEL_VALUES}
        nan_count = total_len - valid_total
        distribution = (
            {label: counts[label] / valid_total for label in LABEL_VALUES} if valid_total else {label: 0.0 for label in LABEL_VALUES}
        )

        report = {
            "symbol": self.symbol,
            "interval": self.interval,
            "backW": backW,
            "forW": forW,
            "alpha": alpha,
            "beta": beta,
            "fee": self.fee,
            "dataset_path": str(dataset_path),
            "counts": {**counts, "NaN": nan_count},
            "distribution": distribution,
            "n_valid": valid_total,
            "n_total": total_len,
        }

        report_base = self.paths.report_base(self.symbol, self.interval, backW, forW)
        report_json = report_base.with_suffix(".json")

        report_json.write_text(json.dumps(report, indent=2, default=str))

class Stage1Pipeline:
    """Stage 1 pipeline orchestrating loader, QA, labeling, and persistence."""

    def __init__(
        self,
        cfg: Stage1Config,
        history_loader: HistoryLoader | None = None,
        quality_checker: 'DataQualityChecker' | None = None,
        return_calculator: 'ForwardReturnCalculator' | None = None,
        calibrator: 'ThresholdCalibrator' | None = None,
        label_assigner: 'LabelAssigner' | None = None,
        window_indexer: 'WindowIndexer' | None = None,
        artifact_writer: 'ArtifactWriter' | None = None,
        ema_calc: 'EMACalculator' | None = None,
    ) -> None:
        self.cfg = cfg
        self.paths = cfg.paths
        self.paths.ensure()
        self.history_loader = history_loader or HistoryLoader()
        self.quality_checker = quality_checker or DataQualityChecker()
        self.return_calculator = return_calculator or ForwardReturnCalculator(cfg.fee)
        self.calibrator = calibrator or ThresholdCalibrator(cfg)
        self.label_assigner = label_assigner or LabelAssigner()
        self.window_indexer = window_indexer or WindowIndexer()
        self.artifact_writer = artifact_writer or ArtifactWriter(self.paths, cfg.symbol, cfg.interval, cfg.fee)
        self.ema_calc = ema_calc or EMACalculator()
        self.df: pd.DataFrame | None = None
        self.qa_report: QAReport | None = None

    def run(self) -> QAReport:
        """
        Orchestrate Stage 1 with "by-the-book" calibration:
          - Prepare global (alpha,beta) from single-candle returns (or constants)
          - Scale beta per forward horizon (beta_forW = beta*(1 + 0.1*(forW-1)))
          - Optionally compute EMA(backW) for completeness (not used in labels)
        """
        df = self.history_loader.load(self.cfg.symbol, self.cfg.interval, start_time=self.cfg.start_date)
        if df.empty:
            raise ValueError(f"No data available for {self.cfg.symbol} {self.cfg.interval}")
        processed_path = self.paths.processed_file(self.cfg.symbol, self.cfg.interval)
        _write_dataframe(df, processed_path, index=True)
        self.df = df

        interval_minutes = interval_to_minutes(self.cfg.interval)
        qa_report = self.quality_checker.evaluate(df, interval_minutes)
        self.qa_report = qa_report
        if qa_report.status == "FAIL":
            raise RuntimeError(f"QA failed: {qa_report}")

        # Prepare global thresholds once (exact_paper mode)
        if self.cfg.calibration_mode == "exact_paper":
            self.calibrator.prepare_global_thresholds(df)

        for forW in self.cfg.forW_values:
            forward = self.return_calculator.compute(df, forW)
            forward_abs = forward.abs().dropna()

            # alpha/beta per paper: ignore forward_abs, use global; per_forw_percentile: use forward_abs
            alpha, beta = self.calibrator.calibrate(forward_abs, forW)
            labels_all = self.label_assigner.assign(forward, alpha, beta)

            for backW in self.cfg.backW_values:
                # Optional: compute EMA(backW) for completeness (not used in labeling rule) will be used later on in stage 2
                ema_col = self.ema_calc.compute(df, backW)
                df_with_ema = df.copy()
                df_with_ema[ema_col.name] = ema_col

                windows = self.window_indexer.build(df_with_ema, backW, forW)
                if windows.empty:
                    continue

                index = windows.index
                sliced_df = df_with_ema.loc[index]
                sliced_labels = labels_all.loc[index]

                self.artifact_writer.write(sliced_df, sliced_labels, backW, forW, alpha, beta)

        return qa_report
