from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from binance.error import ClientError
from binance.spot import Spot


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


def to_ms(ts: int | float | str | datetime | None) -> Optional[int]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)
    return int(pd.to_datetime(ts, utc=True).timestamp() * 1000)


def to_timestamp(ts: int | float | str | datetime | None) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return pd.to_datetime(ts, utc=True)


def _klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if df.empty:
        return df
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for col in float_cols:
        df[col] = df[col].astype(float)
    for col in ["open_time", "close_time", "trade_count"]:
        df[col] = df[col].astype("int64")
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


@dataclass
class HistoryLoaderConfig:
    history_dir: Path = Path("history")
    request_limit: int = 1000
    request_pause: float = 0.2
    pair_config: Optional[Dict[str, Dict[str, Any]]] = None


class HistoryLoader:
    """Upserts Binance kline history and returns normalized OHLCV data."""

    def __init__(
        self,
        client: Spot | None = None,
        config: HistoryLoaderConfig | None = None,
    ) -> None:
        self.client = client or Spot()
        self.config = config or HistoryLoaderConfig()
        self.history_dir = self.config.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: Optional[int | float | str | datetime] = None,
        end_time: Optional[int | float | str | datetime] = None,
        fetch_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Ensure cached history is up to date and return cleaned OHLCV data.

        Args:
            symbol: Binance trading pair, e.g. "BTCUSDT".
            interval: Interval string understood by Binance API, e.g. "1h".
            start_time: Optional inclusive lower bound for returned data.
            end_time: Optional inclusive upper bound for returned data.
            fetch_missing: Whether to call the remote API when cache is stale.

        Returns:
            DataFrame indexed by UTC timestamps with float OHLCV columns.
        """

        raw_df = self._read_cached(symbol, interval)
        if fetch_missing:
            raw_df = self._ensure_remote(symbol, interval, raw_df, start_time, end_time)

        cleaned = self._normalize(raw_df)
        cleaned = self._trim_range(cleaned, start_time, end_time)
        return cleaned

    def ensure(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: Optional[int | float | str | datetime] = None,
        end_time: Optional[int | float | str | datetime] = None,
    ) -> pd.DataFrame:
        """Convenience wrapper that always fetches missing data."""

        return self.load(symbol, interval, start_time=start_time, end_time=end_time, fetch_missing=True)

    def ensure_all(self) -> None:
        """Refresh history for all entries declared in the config."""

        if not self.config.pair_config:
            return
        for symbol, settings in self.config.pair_config.items():
            intervals: Iterable[str] = settings.get("intervals", [])
            start_time = settings.get("start")
            end_time = settings.get("end")
            for interval in intervals:
                try:
                    self.ensure(symbol, interval, start_time=start_time, end_time=end_time)
                except Exception as exc: 
                    print(f"Failed to update {symbol} {interval}: {exc}")

    def _history_path(self, symbol: str, interval: str) -> Path:
        folder = self.history_dir / symbol.upper()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{interval}.csv"

    def _read_cached(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self._history_path(symbol, interval)
        if not path.exists():
            return pd.DataFrame(columns=KLINE_COLS)
        df = pd.read_csv(path)
        if df.empty:
            return df
        if "open_time" in df:
            df["open_time"] = df["open_time"].astype("int64")
        if "close_time" in df:
            df["close_time"] = df["close_time"].astype("int64")
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in numeric_cols:
            if col in df:
                df[col] = df[col].astype(float)
        if "open_dt" in df:
            df["open_dt"] = pd.to_datetime(df["open_dt"], format='ISO8601', utc=True)
        if "close_dt" in df:
            df["close_dt"] = pd.to_datetime(df["close_dt"], format='ISO8601', utc=True)
        return df

    def _save_cache(self, symbol: str, interval: str, df: pd.DataFrame) -> Path:
        path = self._history_path(symbol, interval)
        df.sort_values("open_time").drop_duplicates(subset=["open_time"]).to_csv(path, index=False)
        return path

    def _ensure_remote(
        self,
        symbol: str,
        interval: str,
        cached: pd.DataFrame,
        start_time: Optional[int | float | str | datetime],
        end_time: Optional[int | float | str | datetime],
    ) -> pd.DataFrame:
        start_ms = self._fetch_start_ms(cached, start_time)
        end_ms = to_ms(end_time)
        if start_ms is None and cached.empty:
            start_ms = 0

        if start_ms is None:
            return cached.sort_values("open_time").drop_duplicates(subset=["open_time"])

        missing = self._fetch_missing_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)
        if missing.empty:
            return cached.sort_values("open_time").drop_duplicates(subset=["open_time"])

        combined = (
            pd.concat([cached, missing], ignore_index=True)
            if not cached.empty
            else missing
        )
        combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"])
        self._save_cache(symbol, interval, combined)
        return combined

    def _fetch_start_ms(self, cached: pd.DataFrame, start_time: Optional[int | float | str | datetime]) -> Optional[int]:
        if cached.empty:
            return to_ms(start_time) if start_time is not None else 0
        return int(cached["open_time"].max()) + 1

    def _fetch_missing_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: Optional[int],
    ) -> pd.DataFrame:
        rows: List[List[Any]] = []
        cursor = start_ms
        last_open: Optional[int] = None

        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": self.config.request_limit,
                "startTime": cursor,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            try:
                chunk = self.client.klines(**params)
            except ClientError as exc:  # pragma: no cover - network failure handling
                print(f"{symbol} {interval}: request failed ({exc.error_code}) {exc.error_message}")
                break

            if not chunk:
                break

            if last_open is not None:
                chunk = [row for row in chunk if row[0] > last_open]
                if not chunk:
                    break

            rows.extend(chunk)
            last_open = chunk[-1][0]
            cursor = chunk[-1][6] + 1

            if len(chunk) < self.config.request_limit or (end_ms is not None and cursor > end_ms):
                break

            time.sleep(self.config.request_pause)

        if not rows:
            return pd.DataFrame(columns=KLINE_COLS)

        return _klines_to_df(rows)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            idx = pd.DatetimeIndex([], name="ts")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=idx)

        normalized = df.copy()
        normalized["ts"] = pd.to_datetime(normalized["open_time"], unit="ms", utc=True)
        normalized = normalized.sort_values("ts").drop_duplicates(subset=["ts"])

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        normalized[ohlcv_cols] = normalized[ohlcv_cols].astype(float)

        result = normalized.set_index("ts")[ohlcv_cols]
        result.index.name = "ts"
        return result

    def _trim_range(
        self,
        df: pd.DataFrame,
        start_time: Optional[int | float | str | datetime],
        end_time: Optional[int | float | str | datetime],
    ) -> pd.DataFrame:
        start_ts = to_timestamp(start_time)
        end_ts = to_timestamp(end_time)
        result = df
        if start_ts is not None:
            result = result[result.index >= start_ts]
        if end_ts is not None:
            result = result[result.index <= end_ts]
        return result

