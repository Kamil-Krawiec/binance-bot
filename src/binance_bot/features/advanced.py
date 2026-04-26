from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    return (series - rolling.mean()) / rolling.std()


def _add_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("ts").copy()
    close = result["close"]
    volume = result["volume"]
    log_close = np.log(close)

    result["return_1"] = close.pct_change(1)
    result["return_3"] = close.pct_change(3)
    result["return_6"] = close.pct_change(6)
    result["return_12"] = close.pct_change(12)
    result["log_return_1"] = log_close.diff()
    result["rolling_volatility_6"] = result["log_return_1"].rolling(6, min_periods=6).std()
    result["rolling_volatility_12"] = result["log_return_1"].rolling(12, min_periods=12).std()
    result["rolling_volatility_24"] = result["log_return_1"].rolling(24, min_periods=24).std()
    result["volume_change"] = volume.pct_change()
    result["volume_zscore_24"] = _zscore(volume, 24)
    result["high_low_range_pct"] = (result["high"] - result["low"]) / result["open"]
    result["close_open_return"] = (result["close"] - result["open"]) / result["open"]

    if "ema_20" in result.columns:
        result["distance_to_ema_20_pct"] = (close - result["ema_20"]) / close
    if "ema_50" in result.columns:
        result["distance_to_ema_50_pct"] = (close - result["ema_50"]) / close
    if {"ema_20", "ema_50"}.issubset(result.columns):
        result["ema_20_50_ratio"] = result["ema_20"] / result["ema_50"] - 1
    if {"ema_50", "ema_100"}.issubset(result.columns):
        result["ema_50_100_ratio"] = result["ema_50"] / result["ema_100"] - 1

    rolling_high = result["high"].rolling(24, min_periods=24).max()
    rolling_low = result["low"].rolling(24, min_periods=24).min()
    rolling_close_high = close.rolling(24, min_periods=24).max()
    result["rolling_max_drawdown_24"] = close / rolling_close_high - 1
    result["rolling_high_breakout_24"] = close / rolling_high - 1
    result["rolling_low_breakdown_24"] = close / rolling_low - 1

    if "fng" in result.columns:
        result["fng_change_3d"] = result["fng"] - result["fng"].shift(18)
        result["fng_change_14d"] = result["fng"] - result["fng"].shift(84)
        result["fng_rolling_mean_7d"] = result["fng"].rolling(42, min_periods=42).mean()
        result["fng_rolling_mean_30d"] = result["fng"].rolling(180, min_periods=180).mean()
    if "fng_classification" in result.columns:
        result["fng_extreme_fear"] = (result["fng_classification"] == "Extreme Fear").astype("int64")
        result["fng_extreme_greed"] = (result["fng_classification"] == "Extreme Greed").astype("int64")

    return result


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for advanced features: {missing}")

    frames = [_add_symbol_features(group) for _, group in df.groupby("symbol", sort=False)]
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
