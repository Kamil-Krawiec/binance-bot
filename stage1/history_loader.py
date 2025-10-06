from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from binance.error import ClientError
from binance.spot import Spot

# Configure the datasets you want. Extend the mapping as needed.
PAIR_CONFIG: Dict[str, Dict[str, Any]] = {
    "BTCUSDT": {"intervals": ["1h", "15m"], "start": "2021-01-01 00:00:00 UTC"},
    "ETHUSDT": {"intervals": ["1h", "15m"], "start": "2021-01-01 00:00:00 UTC"},
}

DEFAULT_START_TIME = "2020-01-01 00:00:00 UTC"
REQUEST_LIMIT = 1000
REQUEST_PAUSE = 0.2

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

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

spot = Spot()


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


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if df.empty:
        return df
    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in float_cols:
        df[col] = df[col].astype(float)
    for col in ["open_time", "close_time", "trade_count"]:
        df[col] = df[col].astype("int64")
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def history_path(symbol: str, interval: str) -> Path:
    folder = HISTORY_DIR / symbol.upper()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{interval}.csv"


def load_history(symbol: str, interval: str) -> pd.DataFrame:
    path = history_path(symbol, interval)
    if not path.exists():
        return pd.DataFrame()
    
    
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "open_time" in df:
        df["open_time"] = df["open_time"].astype("int64")
    if "close_time" in df:
        df["close_time"] = df["close_time"].astype("int64")
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        if col in df:
            df[col] = df[col].astype(float)
    if "open_dt" in df:
        df["open_dt"] = pd.to_datetime(df["open_dt"], utc=True)
    if "close_dt" in df:
        df["close_dt"] = pd.to_datetime(df["close_dt"], format='ISO8601', utc=True)
    return df.sort_values("open_time").drop_duplicates(subset=["open_time"])


def save_history(symbol: str, interval: str, df: pd.DataFrame) -> Path:
    path = history_path(symbol, interval)
    df.sort_values("open_time").drop_duplicates(subset=["open_time"]).to_csv(path, index=False)
    return path


def fetch_missing_klines(symbol: str, interval: str, start_ms: int, end_ms: Optional[int] = None) -> pd.DataFrame:
    rows: List[List[Any]] = []
    cursor = start_ms
    last_open: Optional[int] = None

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": REQUEST_LIMIT, "startTime": cursor}
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            chunk = spot.klines(**params)
        except ClientError as exc:
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
        if len(chunk) < REQUEST_LIMIT or (end_ms is not None and cursor > end_ms):
            break
        time.sleep(REQUEST_PAUSE)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLS)

    return klines_to_df(rows)


def ensure_history(symbol: str, interval: str, start_time: Optional[str | int | datetime] = None, end_time: Optional[str | int | datetime] = None) -> None:
    existing = load_history(symbol, interval)
    start_ms = to_ms(start_time) if existing.empty else int(existing["open_time"].max()) + 1
    end_ms = to_ms(end_time)

    if start_ms is None:
        start_ms = 0

    new_df = fetch_missing_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)
    if new_df.empty:
        print(f"{symbol} {interval}: up to date ({len(existing)} rows)")
        return

    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"])
    path = save_history(symbol, interval, combined)
    print(f"{symbol} {interval}: fetched {len(new_df)} rows, total {len(combined)} -> {path}")


def main() -> None:
    for symbol, settings in PAIR_CONFIG.items():
        intervals: Iterable[str] = settings.get("intervals", [])
        start_time = settings.get("start", DEFAULT_START_TIME)
        end_time = settings.get("end")
        for interval in intervals:
            ensure_history(symbol, interval, start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    main()
