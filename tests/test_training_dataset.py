from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from binance_bot.config import (  # noqa: E402
    BalanceConfig,
    TrainingAnalysisConfig,
    TrainingDatasetConfig,
    TrainingFeatureSelectionConfig,
    TrainingSplitConfig,
)
from binance_bot.datasets.training import prepare_training_dataset  # noqa: E402


def _master_fixture(path: Path, periods: int = 520) -> None:
    frames = []
    for symbol, offset in [("AAAUSDT", 0.0), ("BBBUSDT", 20.0)]:
        ts = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
        base = pd.Series(range(periods), dtype="float64")
        labels = ["HOLD", "HOLD", "HOLD", "BUY", "HOLD", "SELL"] * ((periods // 6) + 1)
        labels = labels[:periods]
        frame = pd.DataFrame(
            {
                "ts": ts,
                "symbol": symbol,
                "open": 100 + offset + base * 0.1,
                "high": 101 + offset + base * 0.1,
                "low": 99 + offset + base * 0.1,
                "close": 100.25 + offset + base * 0.1,
                "volume": 1000 + base * 3,
                "label": labels,
                "label_id": [{"HOLD": 0, "BUY": 1, "SELL": 2}[label] for label in labels],
                "backW": 5,
                "forW": 2,
                "alpha": 0.01,
                "beta": 0.05,
                "fee": 0.001,
                "ema_backW": 100 + offset + base * 0.1,
                "RSI": 50 + (base % 10),
                "ULTOSC": 40 + (base % 5),
                "zscore_price": 0.1,
                "zscore_vol": 0.2,
                "pct_change": 0.001,
                "ema_1": 100.25 + offset + base * 0.1,
                "ema_20": 100 + offset + base * 0.1,
                "ema_50": 99 + offset + base * 0.1,
                "ema_100": 98 + offset + base * 0.1,
                "cross_1_20": 0.25,
                "cross_20_50": 1.0,
                "cross_50_100": 1.0,
                "cross_1_50": 1.25,
                "month": ts.month,
                "day_of_week": ts.dayofweek,
                "hour": ts.hour,
                "fng": 30 + (base % 20),
                "fng_classification": ["Fear" if i % 2 else "Neutral" for i in range(periods)],
                "fng_classification_id": [1 if i % 2 else 2 for i in range(periods)],
                "fng_change_1d": 1.0,
                "fng_change_7d": 2.0,
                "fng_is_fear": [1 if i % 2 else 0 for i in range(periods)],
                "fng_is_greed": 0,
            }
        )
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


class TrainingDatasetFunctionalTests(unittest.TestCase):
    def test_prepare_training_dataset_splits_balances_and_records_features(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master_path = root / "master.parquet"
            _master_fixture(master_path)
            config = TrainingDatasetConfig(
                master_path=master_path,
                output_dir=root / "training",
                feature_selection=TrainingFeatureSelectionConfig(
                    include_groups=("technical", "advanced", "time", "sentiment"),
                    exclude_columns=("open", "high", "low", "close", "volume", "ema_1", "ema_20", "ema_50", "ema_100"),
                    leakage_exclude_columns=("return_1", "log_return_1", "pct_change"),
                    lag_periods=1,
                ),
                split=TrainingSplitConfig(validation_cutoff="2024-02-05", validation_end="today"),
                balance=BalanceConfig(mode="undersample_hold", hold_multiplier=1.5, random_state=7),
                analysis=TrainingAnalysisConfig(enabled=False),
            )

            result = prepare_training_dataset(config)
            train = pd.read_parquet(result.train_path)
            validation = pd.read_parquet(result.validation_path)

            self.assertGreater(len(train), 0)
            self.assertGreater(len(validation), 0)
            self.assertGreater(len(result.feature_columns), 10)
            self.assertIn("return_12", result.feature_columns)
            self.assertNotIn("return_1", result.feature_columns)
            self.assertNotIn("log_return_1", result.feature_columns)
            self.assertNotIn("pct_change", result.feature_columns)
            self.assertIn("fng_change_14d", result.feature_columns)
            self.assertNotIn("open", result.feature_columns)
            self.assertLess(train["ts"].max(), pd.Timestamp("2024-02-05", tz="UTC"))
            self.assertGreaterEqual(validation["ts"].min(), pd.Timestamp("2024-02-05", tz="UTC"))
            self.assertEqual(int(train[result.feature_columns].isna().sum().sum()), 0)
            self.assertEqual(int(validation[result.feature_columns].isna().sum().sum()), 0)

            counts = train["label"].value_counts()
            max_action = max(int(counts.get("BUY", 0)), int(counts.get("SELL", 0)))
            self.assertLessEqual(int(counts.get("HOLD", 0)), int(max_action * 1.5))
            self.assertTrue(result.manifest_path.exists())

    def test_prepare_training_dataset_falls_back_when_requested_cutoff_is_after_master_end(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master_path = root / "master.parquet"
            _master_fixture(master_path)
            config = TrainingDatasetConfig(
                master_path=master_path,
                output_dir=root / "training",
                feature_selection=TrainingFeatureSelectionConfig(
                    include_groups=("technical", "advanced", "time", "sentiment"),
                    exclude_columns=("open", "high", "low", "close", "volume", "ema_1", "ema_20", "ema_50", "ema_100"),
                    leakage_exclude_columns=("return_1", "log_return_1", "pct_change"),
                    lag_periods=1,
                ),
                split=TrainingSplitConfig(
                    validation_cutoff="2025-12-30",
                    validation_end="today",
                    fallback_last_available_months=1,
                ),
                balance=BalanceConfig(mode="undersample_hold", hold_multiplier=1.5, random_state=7),
                analysis=TrainingAnalysisConfig(enabled=False),
            )

            result = prepare_training_dataset(config)
            manifest = pd.read_json(result.manifest_path, typ="series")
            self.assertIn("requested_cutoff_after_master_end", manifest["cutoff_fallback_reason"])
            self.assertGreater(result.validation_rows, 0)


if __name__ == "__main__":
    unittest.main()
