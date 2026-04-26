from __future__ import annotations

import importlib.util

import pandas as pd

try:
    import pandas_ta as ta  # noqa: F401
except Exception:  # pragma: no cover - optional dependency fallback is handled by pandas_ta access checks
    ta = None


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).rolling(length, min_periods=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _ultimate_oscillator(df: pd.DataFrame, short: int = 7, medium: int = 14, long: int = 28) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_low = pd.concat([df["low"], previous_close], axis=1).min(axis=1)
    true_high = pd.concat([df["high"], previous_close], axis=1).max(axis=1)
    buying_pressure = df["close"] - true_low
    true_range = true_high - true_low

    def avg(window: int) -> pd.Series:
        return buying_pressure.rolling(window, min_periods=window).sum() / true_range.rolling(window, min_periods=window).sum()

    return 100 * ((4 * avg(short)) + (2 * avg(medium)) + avg(long)) / 7


def _zscore(series: pd.Series, length: int = 30) -> pd.Series:
    rolling = series.rolling(length, min_periods=length)
    return (series - rolling.mean()) / rolling.std()


def _bollinger(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(length, min_periods=length).mean()
    sigma = close.rolling(length, min_periods=length).std()
    lower = middle - std * sigma
    upper = middle + std * sigma
    bandwidth = ((upper - lower) / middle) * 100
    percent = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "BBL_20_2.0": lower,
            "BBM_20_2.0": middle,
            "BBU_20_2.0": upper,
            "BBB_20_2.0": bandwidth,
            "BBP_20_2.0": percent,
        },
        index=close.index,
    )


def _candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if ta is None or not hasattr(df, "ta"):
        return pd.DataFrame(index=df.index)
    if importlib.util.find_spec("talib") is None:
        return pd.DataFrame(index=df.index)
    try:
        patterns = df.ta.cdl_pattern(
            name=["doji", "engulfing", "hammer", "shootingstar", "morningstar", "eveningstar"]
        )
    except Exception:
        return pd.DataFrame(index=df.index)
    if patterns is None:
        return pd.DataFrame(index=df.index)
    return patterns


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the OHLCV-derived features from the research notebook."""
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    result = df.copy()

    if ta is not None and hasattr(result, "ta"):
        result["RSI"] = result.ta.rsi(length=14)
        result["ULTOSC"] = result.ta.uo()
        bb = result.ta.bbands(length=20, std=2)
        if bb is not None:
            result = pd.concat([result, bb], axis=1)
        result["zscore_price"] = result.ta.zscore(close=result["close"], length=30)
        result["zscore_vol"] = result.ta.zscore(close=result["volume"], length=30)
    else:
        result["RSI"] = _rsi(result["close"], length=14)
        result["ULTOSC"] = _ultimate_oscillator(result)
        result = pd.concat([result, _bollinger(result["close"])], axis=1)
        result["zscore_price"] = _zscore(result["close"], length=30)
        result["zscore_vol"] = _zscore(result["volume"], length=30)

    result["pct_change"] = result["close"].pct_change()
    result["ema_1"] = result["close"]
    result["ema_20"] = result["close"].ewm(span=20, adjust=False).mean()
    result["ema_50"] = result["close"].ewm(span=50, adjust=False).mean()
    result["ema_100"] = result["close"].ewm(span=100, adjust=False).mean()
    result["cross_1_20"] = result["ema_1"] - result["ema_20"]
    result["cross_20_50"] = result["ema_20"] - result["ema_50"]
    result["cross_50_100"] = result["ema_50"] - result["ema_100"]
    result["cross_1_50"] = result["ema_1"] - result["ema_50"]
    result["month"] = result.index.month
    result["day_of_week"] = result.index.dayofweek
    result["hour"] = result.index.hour

    patterns = _candlestick_patterns(result)
    if not patterns.empty:
        result = pd.concat([result, patterns], axis=1)

    return result
