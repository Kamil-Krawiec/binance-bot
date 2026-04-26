from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from binance_bot.datasets.master import LABEL_ID_MAP


KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


@dataclass(frozen=True)
class RefreshConfig:
    symbols_file: Path = Path("stage1/labeled_symbols.csv")
    history_dir: Path = Path("stage1/history")
    processed_dir: Path = Path("stage1/stage2_data/processed")
    datasets_dir: Path = Path("stage1/stage2_data/datasets")
    reports_dir: Path = Path("stage1/stage2_data/reports")
    fear_greed_path: Path = Path("cache/fear_greed_daily.csv")
    interval: str = "4h"
    start_time: str = "2025-12-18 08:00:00+00:00"
    end_time: str = "today"
    backW: int = 5
    forW: int = 2
    request_pause: float = 0.05


@dataclass(frozen=True)
class RefreshResult:
    symbols_seen: int
    symbols_refreshed: int
    symbols_failed: dict[str, str] = field(default_factory=dict)
    max_processed_ts: str | None = None
    fear_greed_rows: int = 0


def _to_ms(value: str | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True)
    return int(ts.timestamp() * 1000)


def _end_time(value: str) -> pd.Timestamp:
    if value.lower() == "today":
        return pd.Timestamp.utcnow().floor("4h")
    return pd.to_datetime(value, utc=True)


def _read_symbols(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Symbols file missing symbol column: {path}")
    return sorted(set(df["symbol"].astype(str).str.strip().str.upper()))


def _history_path(history_dir: Path, symbol: str, interval: str) -> Path:
    path = history_dir / symbol / f"{interval}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLS)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=KLINE_COLS)
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int | None) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": cursor}
        if end_ms is not None:
            params["endTime"] = end_ms
        url = f"https://api.binance.com/api/v3/klines?{urlencode(params)}"
        with urlopen(url, timeout=20) as response:
            chunk = json.load(response)
        if not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1][6]) + 1
        if len(chunk) < 1000 or (end_ms is not None and cursor > end_ms):
            break
    if not rows:
        return pd.DataFrame(columns=KLINE_COLS)
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    for col in ["open_time", "close_time", "trade_count"]:
        df[col] = df[col].astype("int64")
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def _normalize_history(raw: pd.DataFrame, start_date: str = "2020-01-01") -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="ts"))
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    df = df[df["ts"] >= pd.to_datetime(start_date, utc=True)]
    ohlcv = ["open", "high", "low", "close", "volume"]
    df[ohlcv] = df[ohlcv].astype(float)
    result = df.set_index("ts")[ohlcv].sort_index()
    result.index.name = "ts"
    return result


def _load_report_thresholds(reports_dir: Path, symbol: str, interval: str, backW: int, forW: int) -> tuple[float, float, float]:
    path = reports_dir / f"{symbol}_{interval}_backW{backW}_forW{forW}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["alpha"]), float(payload["beta"]), float(payload["fee"])


def _labels_for(df: pd.DataFrame, *, backW: int, forW: int, alpha: float, beta: float, fee: float) -> pd.DataFrame:
    future_close = df["close"].shift(-forW)
    forward = ((1.0 - fee) * future_close - (1.0 + fee) * df["open"]) / df["open"]
    labels = pd.Series("HOLD", index=df.index, dtype="object")
    labels.loc[forward.isna()] = pd.NA
    valid = forward.notna()
    labels.loc[(forward > alpha) & (forward < beta) & valid] = "BUY"
    labels.loc[(forward < -alpha) & (forward > -beta) & valid] = "SELL"
    ema = df["close"].ewm(span=backW, adjust=False).mean()

    start = backW - 1
    end = len(df) - forW
    idx = df.index[start:end]
    result = pd.DataFrame(
        {
            "ts": idx,
            "label": labels.loc[idx].astype("string").to_numpy(),
            "backW": backW,
            "forW": forW,
            "alpha": alpha,
            "beta": beta,
            "fee": fee,
            "ema_backW": ema.loc[idx].to_numpy(),
        }
    )
    return result


def _write_report(
    reports_dir: Path,
    dataset_path: Path,
    *,
    symbol: str,
    interval: str,
    backW: int,
    forW: int,
    labels: pd.DataFrame,
    alpha: float,
    beta: float,
    fee: float,
) -> None:
    counts = {label: int((labels["label"] == label).sum()) for label in LABEL_ID_MAP}
    valid_total = int(labels["label"].count())
    distribution = {label: counts[label] / valid_total for label in LABEL_ID_MAP} if valid_total else {}
    payload = {
        "symbol": symbol,
        "interval": interval,
        "backW": backW,
        "forW": forW,
        "alpha": alpha,
        "beta": beta,
        "fee": fee,
        "dataset_path": str(dataset_path.resolve()),
        "counts": counts,
        "distribution": distribution,
        "n_valid": valid_total,
        "n_total": int(len(labels)),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{symbol}_{interval}_backW{backW}_forW{forW}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def refresh_fear_greed(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen("https://api.alternative.me/fng/?limit=0&format=json", timeout=30) as response:
        payload = json.load(response)
    rows = []
    for item in payload.get("data", []):
        rows.append(
            {
                "date": pd.to_datetime(int(item["timestamp"]), unit="s", utc=True).date().isoformat(),
                "fng": int(item["value"]),
                "classification": item["value_classification"],
            }
        )
    df = pd.DataFrame(rows).sort_values("date", ascending=False)
    df.to_csv(path, index=False)
    return len(df)


def refresh_history_and_labels(config: RefreshConfig) -> RefreshResult:
    symbols = _read_symbols(config.symbols_file)
    end_ts = _end_time(config.end_time)
    end_ms = _to_ms(end_ts)
    min_start_ms = _to_ms(config.start_time)
    failures: dict[str, str] = {}
    refreshed = 0
    max_processed_ts: pd.Timestamp | None = None

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.datasets_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    for index, symbol in enumerate(symbols, start=1):
        path = _history_path(config.history_dir, symbol, config.interval)
        try:
            raw = _read_history(path)
            cached_next = int(raw["open_time"].max()) + 1 if not raw.empty else min_start_ms
            start_ms = max(cached_next, min_start_ms or cached_next)
            if start_ms is not None and end_ms is not None and start_ms <= end_ms:
                missing = _fetch_klines(symbol, config.interval, start_ms, end_ms)
                if not missing.empty:
                    raw = pd.concat([raw, missing], ignore_index=True)
                    raw = raw.sort_values("open_time").drop_duplicates(subset=["open_time"])
                    raw.to_csv(path, index=False)
            processed = _normalize_history(raw)
            if processed.empty:
                failures[symbol] = "empty_processed"
                continue
            processed_path = config.processed_dir / f"{symbol}_{config.interval}.parquet"
            processed.to_parquet(processed_path)
            alpha, beta, fee = _load_report_thresholds(
                config.reports_dir, symbol, config.interval, config.backW, config.forW
            )
            labels = _labels_for(processed, backW=config.backW, forW=config.forW, alpha=alpha, beta=beta, fee=fee)
            dataset_path = config.datasets_dir / f"{symbol}_{config.interval}_backW{config.backW}_forW{config.forW}.parquet"
            labels.to_parquet(dataset_path, index=False)
            _write_report(
                config.reports_dir,
                dataset_path,
                symbol=symbol,
                interval=config.interval,
                backW=config.backW,
                forW=config.forW,
                labels=labels,
                alpha=alpha,
                beta=beta,
                fee=fee,
            )
            max_processed_ts = processed.index.max() if max_processed_ts is None else max(max_processed_ts, processed.index.max())
            refreshed += 1
        except (HTTPError, URLError, TimeoutError, FileNotFoundError, ValueError) as exc:
            failures[symbol] = str(exc)
        if index % 25 == 0:
            print(f"refreshed {index}/{len(symbols)} symbols")
        time.sleep(config.request_pause)

    fear_rows = refresh_fear_greed(config.fear_greed_path)
    return RefreshResult(
        symbols_seen=len(symbols),
        symbols_refreshed=refreshed,
        symbols_failed=failures,
        max_processed_ts=max_processed_ts.isoformat() if max_processed_ts is not None else None,
        fear_greed_rows=fear_rows,
    )
