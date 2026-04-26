from __future__ import annotations

import argparse
import sys
from pathlib import Path

from binance_bot.config import load_master_dataset_config
from binance_bot.datasets.master import build_master_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ML-ready crypto master datasets")
    parser.add_argument(
        "command",
        nargs="?",
        default="build-master-dataset",
        choices=["build-master-dataset"],
        help="Command to run",
    )
    parser.add_argument(
        "--config",
        default="configs/master_dataset.yaml",
        help="Path to master dataset YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "build-master-dataset"
    if command != "build-master-dataset":
        parser.error(f"Unknown command: {command}")

    try:
        config = load_master_dataset_config(Path(args.config))
        result = build_master_dataset(config)
    except Exception as exc:
        print(f"Failed to build master dataset: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {result.rows:,} rows for {len(result.symbols)} symbols -> {result.output_path}")
    if result.skipped:
        print("Skipped:")
        for key, reason in sorted(result.skipped.items()):
            print(f"  {key}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
