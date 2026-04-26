from __future__ import annotations

import argparse
import sys
from pathlib import Path

from binance_bot.config import load_eda_config, load_master_dataset_config, load_training_dataset_config
from binance_bot.datasets.master import build_master_dataset
from binance_bot.datasets.refresh import RefreshConfig, refresh_history_and_labels
from binance_bot.datasets.training import prepare_training_dataset
from binance_bot.eda.analysis import run_eda_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ML-ready crypto master datasets")
    parser.add_argument(
        "command",
        nargs="?",
        default="build-master-dataset",
        choices=["build-master-dataset", "prepare-training-dataset", "refresh-b5-f2-data", "run-eda-analysis"],
        help="Command to run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to master dataset YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "build-master-dataset"
    try:
        if command == "build-master-dataset":
            config = load_master_dataset_config(Path(args.config or "configs/master_dataset.yaml"))
            result = build_master_dataset(config)
            print(f"Saved {result.rows:,} rows for {len(result.symbols)} symbols -> {result.output_path}")
            if result.skipped:
                print("Skipped:")
                for key, reason in sorted(result.skipped.items()):
                    print(f"  {key}: {reason}")
        elif command == "prepare-training-dataset":
            config = load_training_dataset_config(Path(args.config or "configs/training_dataset.yaml"))
            result = prepare_training_dataset(config)
            print(f"Saved train rows={result.train_rows:,} -> {result.train_path}")
            print(f"Saved validation rows={result.validation_rows:,} -> {result.validation_path}")
            print(f"Saved manifest -> {result.manifest_path}")
        elif command == "refresh-b5-f2-data":
            result = refresh_history_and_labels(RefreshConfig())
            print(
                f"Refreshed {result.symbols_refreshed}/{result.symbols_seen} symbols; "
                f"max processed ts={result.max_processed_ts}; fear/greed rows={result.fear_greed_rows}"
            )
            if result.symbols_failed:
                print("Failures:")
                for symbol, reason in sorted(result.symbols_failed.items()):
                    print(f"  {symbol}: {reason}")
        elif command == "run-eda-analysis":
            config = load_eda_config(Path(args.config or "configs/eda_analysis.yaml"))
            result = run_eda_analysis(config)
            print(f"Saved EDA overview -> {result.overview_path}")
            print(f"Saved {len(result.table_paths)} EDA tables under {result.output_dir / 'tables'}")
            print(f"Saved {len(result.chart_paths)} EDA chart/library artifacts under {result.output_dir}")
        else:
            parser.error(f"Unknown command: {command}")
    except Exception as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return 1
    return 0


def build_master_main(argv: list[str] | None = None) -> int:
    return main(["build-master-dataset", *(sys.argv[1:] if argv is None else argv)])


def prepare_training_main(argv: list[str] | None = None) -> int:
    return main(["prepare-training-dataset", *(sys.argv[1:] if argv is None else argv)])


def run_eda_main(argv: list[str] | None = None) -> int:
    return main(["run-eda-analysis", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
