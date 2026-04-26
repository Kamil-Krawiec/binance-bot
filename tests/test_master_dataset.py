from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from binance_bot.config import FearGreedConfig, FeatureConfig, MasterDatasetConfig
from binance_bot.datasets.master import build_master_dataset


def _write_symbol_fixture(root: Path, symbol: str, *, periods: int = 140) -> None:
    ts = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
    base = pd.Series(range(periods), index=ts, dtype="float64")
    price = pd.DataFrame(
        {
            "open": 100 + base * 0.5,
            "high": 101 + base * 0.5,
            "low": 99 + base * 0.5,
            "close": 100.25 + base * 0.5,
            "volume": 1000 + base * 10,
        },
        index=ts,
    )
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    price.to_parquet(processed / f"{symbol}_4h.parquet")

    labels = pd.DataFrame(
        {
            "ts": ts[4:-2],
            "label": (["HOLD", "BUY", "SELL"] * periods)[: len(ts[4:-2])],
            "backW": 5,
            "forW": 2,
            "alpha": 0.01,
            "beta": 0.05,
            "fee": 0.001,
            "ema_backW": price["close"].ewm(span=5, adjust=False).mean().loc[ts[4:-2]].to_numpy(),
        }
    )
    labeled = root / "labeled"
    labeled.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labeled / f"{symbol}_4h_backW5_forW2.parquet", index=False)


def _config(root: Path, *, fear_greed_path: Path | None) -> MasterDatasetConfig:
    symbols_file = root / "symbols.csv"
    pd.DataFrame({"symbol": ["AAAUSDT", "BBBUSDT"]}).to_csv(symbols_file, index=False)
    return MasterDatasetConfig(
        backW=5,
        forW=2,
        processed_dir=root / "processed",
        labeled_datasets_dir=root / "labeled",
        symbols_file=symbols_file,
        output_path=root / "masters" / "master.parquet",
        features=FeatureConfig(
            technical=True,
            fear_greed=FearGreedConfig(enabled=True, path=fear_greed_path),
        ),
    )


class MasterDatasetFunctionalTests(unittest.TestCase):
    def test_build_master_dataset_processes_multiple_symbols_and_required_columns(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_symbol_fixture(root, "AAAUSDT")
            _write_symbol_fixture(root, "BBBUSDT")
            fear_path = root / "fear_greed.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=30, freq="D").date,
                    "fng": list(range(30)),
                    "classification": ["Fear"] * 30,
                }
            ).to_csv(fear_path, index=False)

            result = build_master_dataset(_config(root, fear_greed_path=fear_path))
            df = pd.read_parquet(result.output_path)

            self.assertEqual(set(df["symbol"]), {"AAAUSDT", "BBBUSDT"})
            self.assertTrue(
                {"ts", "symbol", "label", "label_id", "backW", "forW", "alpha", "beta", "fee"}.issubset(df.columns)
            )
            self.assertTrue(
                {"RSI", "ULTOSC", "pct_change", "zscore_price", "zscore_vol", "cross_1_20", "hour"}.issubset(
                    df.columns
                )
            )
            self.assertTrue(
                {"fng", "fng_classification_id", "fng_change_1d", "fng_change_7d", "fng_is_fear"}.issubset(df.columns)
            )
            self.assertEqual(int(df.duplicated(subset=["symbol", "ts"]).sum()), 0)
            self.assertEqual(int(df.select_dtypes(include="number").isna().sum().sum()), 0)

    def test_build_master_dataset_allows_missing_optional_fear_greed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_symbol_fixture(root, "AAAUSDT")
            config = _config(root, fear_greed_path=root / "missing.csv")

            result = build_master_dataset(config)
            df = pd.read_parquet(result.output_path)

            self.assertGreater(len(df), 0)
            self.assertNotIn("fng", df.columns)
            self.assertTrue(result.skipped["fear_greed"].startswith("missing:"))


if __name__ == "__main__":
    unittest.main()
