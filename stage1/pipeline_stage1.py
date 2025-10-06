from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd

from history_loader import load_history


Label = Literal["BUY", "HOLD", "SELL"]
LABEL_VALUES: List[Label] = ["BUY", "HOLD", "SELL"]


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
    """Immutable configuration bundle for Stage 1 labeling pipeline."""

    symbol: str
    interval: str
    fee: float
    backW_values: List[int]
    forW_values: List[int]
    percentiles_by_forW: Dict[int, Dict[str, float]]
    paths: Stage1Paths


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


class OHLCVLoader:
    """Load and normalize OHLCV history produced by history_loader."""

    def __init__(self, symbol: str, interval: str, paths: Stage1Paths) -> None:
        self.symbol = symbol
        self.interval = interval
        self.paths = paths

    def load(self) -> pd.DataFrame:
        raw_df = load_history(self.symbol, self.interval)
        if raw_df.empty:
            raise ValueError(
                f"No historical data found for {self.symbol} {self.interval}. Run history_loader first."
            )

        if "open_dt" in raw_df.columns:
            ts = pd.to_datetime(raw_df["open_dt"], utc=True)
        elif "open_time" in raw_df.columns:
            ts = pd.to_datetime(raw_df["open_time"], unit="ms", utc=True)
        else:
            raise ValueError("History must include either 'open_dt' or 'open_time' column")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        clean_df = pd.DataFrame(
            {
                "ts": ts,
                "open": raw_df["open"].astype(float),
                "high": raw_df["high"].astype(float),
                "low": raw_df["low"].astype(float),
                "close": raw_df["close"].astype(float),
                "volume": raw_df["volume"].astype(float),
            }
        )
        clean_df = (
            clean_df.dropna(subset=["ts"])
            .drop_duplicates(subset=["ts"], keep="last")
            .sort_values("ts")
            .set_index("ts")
        )
        processed_path = self.paths.processed_file(self.symbol, self.interval)
        _write_dataframe(clean_df, processed_path, index=True)
        return clean_df


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
    """Compute forward returns for labeling."""

    def __init__(self, fee: float) -> None:
        self.fee = fee

    def compute(self, df: pd.DataFrame, forW: int) -> pd.Series:
        if forW < 0:
            raise ValueError("forW must be non-negative")
        if df.empty:
            return pd.Series(dtype="float64")
        future_close = df["close"].shift(-forW)
        forward = ((1 - self.fee) * future_close - (1 + self.fee) * df["open"]) / df["open"]
        forward.name = f"forward_{forW}"
        return forward


class ThresholdCalibrator:
    """Pick alpha/beta thresholds from absolute forward returns."""

    def __init__(self, percentiles_by_forW: Dict[int, Dict[str, float]]) -> None:
        self.percentiles_by_forW = percentiles_by_forW

    def calibrate(self, forward_abs: pd.Series, forW: int) -> Tuple[float, float]:
        params = self.percentiles_by_forW.get(forW)
        if params is None:
            raise KeyError(f"Percentiles not configured for forW={forW}")
        if forward_abs.empty:
            raise ValueError("Forward returns series is empty for calibration")

        alpha_pct = params.get("alpha_pct")
        beta_pct = params.get("beta_pct")
        if alpha_pct is None or beta_pct is None:
            raise KeyError(f"alpha_pct/beta_pct missing for forW={forW}")

        alpha = float(forward_abs.quantile(alpha_pct))
        beta = float(forward_abs.quantile(beta_pct))
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
        if backW < 1:
            raise ValueError("backW must be >= 1")
        if forW < 0:
            raise ValueError("forW must be >= 0")

        n = len(df)
        if n == 0:
            empty_index = pd.DatetimeIndex([])
            return pd.DataFrame(columns=["t", "backW", "forW"], index=empty_index)

        start = max(backW - 1, 0)
        end = n if forW == 0 else n - forW
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
        report_md = report_base.with_suffix(".md")

        report_json.write_text(json.dumps(report, indent=2, default=str))

        label_lines = [
            f"- BUY: {counts['BUY']}",
            f"- HOLD: {counts['HOLD']}",
            f"- SELL: {counts['SELL']}",
            f"- NaN: {nan_count}",
        ]
        distribution_lines = [f"- {label}: {distribution[label]:.4f}" for label in LABEL_VALUES]

        report_md.write_text(
            "\n".join(
                [
                    f"# Stage 1 Dataset {self.symbol} {self.interval} backW={backW} forW={forW}",
                    "",
                    f"- dataset: {dataset_path}",
                    f"- alpha: {alpha:.6f}",
                    f"- beta: {beta:.6f}",
                    f"- fee: {self.fee:.6f}",
                    "",
                    "## Label Counts",
                    *label_lines,
                    "",
                    "## Distribution",
                    *distribution_lines,
                ]
            )
        )


class Stage1Pipeline:
    """Stage 1 pipeline orchestrating loader, QA, labeling, and persistence."""

    def __init__(
        self,
        cfg: Stage1Config,
        loader: OHLCVLoader | None = None,
        quality_checker: DataQualityChecker | None = None,
        return_calculator: ForwardReturnCalculator | None = None,
        calibrator: ThresholdCalibrator | None = None,
        label_assigner: LabelAssigner | None = None,
        window_indexer: WindowIndexer | None = None,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        self.cfg = cfg
        self.paths = cfg.paths
        self.paths.ensure()

        self.loader = loader or OHLCVLoader(cfg.symbol, cfg.interval, self.paths)
        self.quality_checker = quality_checker or DataQualityChecker()
        self.return_calculator = return_calculator or ForwardReturnCalculator(cfg.fee)
        self.calibrator = calibrator or ThresholdCalibrator(cfg.percentiles_by_forW)
        self.label_assigner = label_assigner or LabelAssigner()
        self.window_indexer = window_indexer or WindowIndexer()
        self.artifact_writer = artifact_writer or ArtifactWriter(self.paths, cfg.symbol, cfg.interval, cfg.fee)

        self.df: pd.DataFrame | None = None
        self.qa_report: QAReport | None = None

    def run(self) -> QAReport:
        df = self.loader.load()
        self.df = df

        interval_minutes = interval_to_minutes(self.cfg.interval)
        qa_report = self.quality_checker.evaluate(df, interval_minutes)
        self.qa_report = qa_report
        if qa_report.status == "FAIL":
            raise RuntimeError(f"QA failed: {qa_report}")

        for forW in self.cfg.forW_values:
            forward = self.return_calculator.compute(df, forW)
            forward_abs = forward.abs().dropna()
            if forward_abs.empty:
                continue
            alpha, beta = self.calibrator.calibrate(forward_abs, forW)
            labels_all = self.label_assigner.assign(forward, alpha, beta)

            for backW in self.cfg.backW_values:
                windows = self.window_indexer.build(df, backW, forW)
                if windows.empty:
                    continue
                index = windows.index
                sliced_df = df.loc[index]
                sliced_labels = labels_all.loc[index]
                self.artifact_writer.write(sliced_df, sliced_labels, backW, forW, alpha, beta)

        return qa_report
