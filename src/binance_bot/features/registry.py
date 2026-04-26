from __future__ import annotations

from collections.abc import Iterable


IDENTIFIER_COLUMNS = ("ts", "symbol")
TARGET_COLUMNS = ("label", "label_id")
LABEL_METADATA_COLUMNS = ("backW", "forW", "alpha", "beta", "fee", "ema_backW")
RAW_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TECHNICAL_COLUMNS = (
    "RSI",
    "ULTOSC",
    "BBL_20_2.0_2.0",
    "BBM_20_2.0_2.0",
    "BBU_20_2.0_2.0",
    "BBB_20_2.0_2.0",
    "BBP_20_2.0_2.0",
    "zscore_price",
    "zscore_vol",
    "pct_change",
    "ema_1",
    "ema_20",
    "ema_50",
    "ema_100",
    "cross_1_20",
    "cross_20_50",
    "cross_50_100",
    "cross_1_50",
)
TIME_COLUMNS = ("month", "day_of_week", "hour")
SENTIMENT_COLUMNS = (
    "fng",
    "fng_classification_id",
    "fng_change_1d",
    "fng_change_3d",
    "fng_change_7d",
    "fng_change_14d",
    "fng_rolling_mean_7d",
    "fng_rolling_mean_30d",
    "fng_is_fear",
    "fng_is_greed",
    "fng_extreme_fear",
    "fng_extreme_greed",
)
ADVANCED_COLUMNS = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "log_return_1",
    "rolling_volatility_6",
    "rolling_volatility_12",
    "rolling_volatility_24",
    "volume_change",
    "volume_zscore_24",
    "high_low_range_pct",
    "close_open_return",
    "distance_to_ema_20_pct",
    "distance_to_ema_50_pct",
    "ema_20_50_ratio",
    "ema_50_100_ratio",
    "rolling_max_drawdown_24",
    "rolling_high_breakout_24",
    "rolling_low_breakdown_24",
)

FEATURE_GROUPS = {
    "identifier": IDENTIFIER_COLUMNS,
    "target": TARGET_COLUMNS,
    "label_metadata": LABEL_METADATA_COLUMNS,
    "raw_ohlcv": RAW_OHLCV_COLUMNS,
    "technical": TECHNICAL_COLUMNS,
    "advanced": ADVANCED_COLUMNS,
    "time": TIME_COLUMNS,
    "sentiment": SENTIMENT_COLUMNS,
}

NON_MODEL_COLUMNS = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS + LABEL_METADATA_COLUMNS + ("fng_classification",))


def columns_for_groups(groups: Iterable[str], available_columns: Iterable[str]) -> list[str]:
    available = set(available_columns)
    columns: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group}")
        for column in FEATURE_GROUPS[group]:
            if column in available and column not in columns:
                columns.append(column)

    for column in available:
        if column.startswith("CDL_") and "technical" in groups and column not in columns:
            columns.append(column)
    return columns
