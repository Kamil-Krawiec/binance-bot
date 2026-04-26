from __future__ import annotations

from pathlib import Path

import pandas as pd


FNG_CLASSIFICATION_MAP = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


def load_fear_greed_features(path: str | Path) -> pd.DataFrame:
    """Load daily Fear & Greed data and derive simple market-context features."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)
    required = {"date", "fng", "classification"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Fear & Greed CSV missing columns: {missing}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.normalize()
    result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    result["fng"] = pd.to_numeric(result["fng"], errors="coerce")
    result["fng_classification_id"] = result["classification"].map(FNG_CLASSIFICATION_MAP).fillna(-1).astype("int64")
    result["fng_change_1d"] = result["fng"].diff()
    result["fng_change_7d"] = result["fng"] - result["fng"].shift(7)
    result["fng_is_fear"] = result["classification"].isin(["Extreme Fear", "Fear"]).astype("int64")
    result["fng_is_greed"] = result["classification"].isin(["Greed", "Extreme Greed"]).astype("int64")
    result = result.rename(columns={"classification": "fng_classification"})
    return result.set_index("date")


class ExogenousFeatureProvider:
    """Extension point for external market-context features."""

    def features_for(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        return frame


class FearGreedFeatureProvider(ExogenousFeatureProvider):
    def __init__(self, path: str | Path | None, *, enabled: bool = True) -> None:
        self.path = Path(path).expanduser() if path else None
        self.enabled = enabled
        self.skipped_reason: str | None = None
        self._features: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame | None:
        if not self.enabled:
            self.skipped_reason = "disabled"
            return None
        if self.path is None:
            self.skipped_reason = "path_not_configured"
            return None
        if not self.path.exists():
            self.skipped_reason = f"missing:{self.path}"
            return None
        if self._features is None:
            self._features = load_fear_greed_features(self.path)
        return self._features

    def features_for(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        features = self._load()
        if features is None:
            return frame

        result = frame.copy()
        join_key = pd.to_datetime(result.index, utc=True).normalize()
        joined = features.reindex(join_key)
        joined.index = result.index
        return pd.concat([result, joined], axis=1)
