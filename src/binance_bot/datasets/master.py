from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from binance_bot.config import MasterDatasetConfig
from binance_bot.features.exogenous import ExogenousFeatureProvider, FearGreedFeatureProvider
from binance_bot.features.technical import add_technical_features


LABEL_ID_MAP = {"HOLD": 0, "BUY": 1, "SELL": 2}
REQUIRED_OUTPUT_COLUMNS = {"ts", "symbol", "label", "label_id", "backW", "forW", "alpha", "beta", "fee"}


@dataclass(frozen=True)
class MasterDatasetResult:
    output_path: Path
    rows: int
    symbols: list[str]
    skipped: dict[str, str] = field(default_factory=dict)


def parse_symbol_interval(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected dataset filename: {path.name}")
    return parts[0].upper(), parts[1]


def load_allowed_symbols(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Symbols file missing 'symbol' column: {path}")
    return set(df["symbol"].astype(str).str.strip().str.upper())


def load_price(processed_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    path = processed_dir / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"
    return df.sort_index()


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"Label dataset missing 'ts': {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def candidate_label_files(labeled_dir: Path, backW: int, forW: int) -> list[Path]:
    return sorted(labeled_dir.glob(f"*_backW{backW}_forW{forW}.parquet"))


def _drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    protected = {"ts", "symbol", "label", "label_id", "backW", "forW", "alpha", "beta", "fee"}
    feature_columns = [
        col
        for col in df.columns
        if col not in protected and not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col])
    ]
    if not feature_columns:
        return df
    return df.dropna(subset=feature_columns)


def build_symbol_dataset(
    label_file: Path,
    *,
    processed_dir: Path,
    backW: int,
    forW: int,
    technical_features: bool,
    exogenous_providers: Iterable[ExogenousFeatureProvider],
) -> pd.DataFrame:
    symbol, interval = parse_symbol_interval(label_file)
    labels = load_labels(label_file)
    prices = load_price(processed_dir, symbol, interval)
    joined = prices.join(labels, how="inner", validate="one_to_one")
    if joined.empty:
        return joined

    window_mismatch = (joined["backW"] != backW) | (joined["forW"] != forW)
    if bool(window_mismatch.any()):
        raise ValueError(f"Window mismatch in {label_file.name}")

    if technical_features:
        joined = add_technical_features(joined)

    for provider in exogenous_providers:
        joined = provider.features_for(joined, symbol)

    joined["label"] = joined["label"].astype("string")
    joined["label_id"] = joined["label"].map(LABEL_ID_MAP).astype("int64")
    joined = _drop_warmup_rows(joined)
    if joined.empty:
        return joined

    joined = joined.reset_index()
    joined["symbol"] = symbol
    return joined


def build_master_dataset(config: MasterDatasetConfig) -> MasterDatasetResult:
    allowed_symbols = load_allowed_symbols(config.symbols_file)
    files = candidate_label_files(config.labeled_datasets_dir, config.backW, config.forW)
    if allowed_symbols is not None:
        files = [path for path in files if parse_symbol_interval(path)[0] in allowed_symbols]
    if not files:
        raise FileNotFoundError(
            f"No label files found for backW={config.backW}, forW={config.forW} in {config.labeled_datasets_dir}"
        )

    fear_greed = FearGreedFeatureProvider(
        config.features.fear_greed.path,
        enabled=config.features.fear_greed.enabled,
    )
    providers: list[ExogenousFeatureProvider] = [fear_greed]

    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for label_file in files:
        symbol, _ = parse_symbol_interval(label_file)
        try:
            frame = build_symbol_dataset(
                label_file,
                processed_dir=config.processed_dir,
                backW=config.backW,
                forW=config.forW,
                technical_features=config.features.technical,
                exogenous_providers=providers,
            )
        except FileNotFoundError as exc:
            skipped[symbol] = f"missing_input:{exc}"
            continue
        except Exception as exc:
            skipped[symbol] = f"failed:{exc}"
            continue
        if frame.empty:
            skipped[symbol] = "empty_after_join_or_features"
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No symbols produced rows. Skipped: {skipped}")

    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["ts", "symbol"]).reset_index(drop=True)
    missing_required = REQUIRED_OUTPUT_COLUMNS - set(master.columns)
    if missing_required:
        raise RuntimeError(f"Master dataset missing required columns: {sorted(missing_required)}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(config.output_path, index=False)

    if fear_greed.skipped_reason:
        skipped["fear_greed"] = fear_greed.skipped_reason

    symbols = sorted(master["symbol"].dropna().astype(str).unique().tolist())
    return MasterDatasetResult(output_path=config.output_path, rows=len(master), symbols=symbols, skipped=skipped)
