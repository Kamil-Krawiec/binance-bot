from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from binance_bot.config import EDAAnalysisConfig, EDAConfig, EDASampleConfig  # noqa: E402
from binance_bot.eda.analysis import run_eda_analysis  # noqa: E402


def _eda_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    feature_columns = [
        "RSI",
        "rolling_volatility_24",
        "high_low_range_pct",
        "fng_is_fear",
        "hour",
        "volume_zscore",
    ]
    frames: dict[str, pd.DataFrame] = {}
    for split, start, periods, drift in [
        ("train", "2025-01-01", 72, 0.0),
        ("validation", "2025-02-01", 36, 0.4),
    ]:
        split_frames = []
        for symbol, offset in [("AAAUSDT", 0.0), ("BBBUSDT", 10.0)]:
            ts = pd.date_range(start, periods=periods, freq="4h", tz="UTC")
            base = pd.Series(range(periods), dtype="float64")
            labels = ["HOLD", "BUY", "SELL", "HOLD", "BUY", "HOLD"] * ((periods // 6) + 1)
            labels = labels[:periods]
            label_id = [{"HOLD": 0, "BUY": 1, "SELL": 2}[label] for label in labels]
            split_frames.append(
                pd.DataFrame(
                    {
                        "ts": ts,
                        "symbol": symbol,
                        "label": labels,
                        "label_id": label_id,
                        "RSI": 40 + offset + drift + (base % 20),
                        "rolling_volatility_24": 0.01 + drift + (base % 9) / 100,
                        "high_low_range_pct": 0.02 + (base % 7) / 100,
                        "fng_is_fear": [1 if i % 3 == 0 else 0 for i in range(periods)],
                        "hour": ts.hour,
                        "volume_zscore": ((base % 11) - 5) / 3,
                    }
                )
            )
        frames[split] = pd.concat(split_frames, ignore_index=True)

    train_path = root / "train.parquet"
    validation_path = root / "validation.parquet"
    manifest_path = root / "feature_manifest.json"
    xgb_path = root / "xgboost_importance.csv"
    frames["train"].to_parquet(train_path, index=False)
    frames["validation"].to_parquet(validation_path, index=False)
    manifest_path.write_text(json.dumps({"feature_columns": feature_columns}), encoding="utf-8")
    pd.DataFrame(
        {
            "feature": ["rolling_volatility_24", "high_low_range_pct", "RSI"],
            "importance_score": [5.0, 3.0, 1.0],
        }
    ).to_csv(xgb_path, index=False)
    return train_path, validation_path, manifest_path, xgb_path


class EDAAnalysisFunctionalTests(unittest.TestCase):
    def test_run_eda_analysis_writes_tables_charts_and_scaler_recommendations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path, validation_path, manifest_path, xgb_path = _eda_fixture(root)
            config = EDAConfig(
                train_path=train_path,
                validation_path=validation_path,
                manifest_path=manifest_path,
                xgboost_importance_path=xgb_path,
                output_dir=root / "eda",
                sample=EDASampleConfig(mutual_info_rows=100, plot_rows=100, library_rows=50),
                analysis=EDAAnalysisConfig(
                    top_n=5,
                    top_distribution_features=3,
                    top_drift_features=3,
                    correlation_threshold=0.9,
                    library_enabled=False,
                ),
            )

            result = run_eda_analysis(config)

            self.assertTrue(result.overview_path.exists())
            self.assertGreaterEqual(len(result.table_paths), 6)
            self.assertTrue(all(path.exists() for path in result.table_paths))
            self.assertGreaterEqual(len(result.chart_paths), 1)
            self.assertTrue(any(path.name == "target_distribution_by_split.png" for path in result.chart_paths))

            overview = json.loads(result.overview_path.read_text(encoding="utf-8"))
            self.assertEqual(overview["train"]["symbols"], 2)
            self.assertEqual(overview["validation"]["symbols"], 2)
            self.assertEqual(overview["feature_count"], 6)

            influence = pd.read_csv(config.output_dir / "tables" / "target_influence.csv")
            self.assertTrue(
                {"feature", "pearson_corr_label_id", "mutual_info_label_id", "xgboost_importance_score"}.issubset(
                    influence.columns
                )
            )

            recommendations = pd.read_csv(config.output_dir / "tables" / "normalization_recommendations.csv")
            by_feature = recommendations.set_index("feature")
            self.assertEqual(by_feature.loc["RSI", "tree_models"], "no_scaling_required")
            self.assertIn(by_feature.loc["RSI", "mlp_logistic"], {"StandardScaler", "RobustScaler_candidate"})
            self.assertEqual(by_feature.loc["fng_is_fear", "mlp_logistic"], "passthrough_binary")
            self.assertIn("hour", by_feature.index)


if __name__ == "__main__":
    unittest.main()
