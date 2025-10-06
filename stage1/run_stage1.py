from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from history_loader import ensure_history  

from pipeline_stage1 import (
    Stage1Config,
    Stage1Paths,
    Stage1Pipeline,
)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file into a Python dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def build_paths_config(raw: Dict[str, Any]) -> Stage1Paths:
    """Create Stage1Paths from YAML 'paths' section."""
    try:
        processed = Path(raw["processed_dir"]).expanduser()
        datasets = Path(raw["datasets_dir"]).expanduser()
        reports = Path(raw["reports_dir"]).expanduser()
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing '{missing}' entry in paths config") from exc
    return Stage1Paths(processed_dir=processed, datasets_dir=datasets, reports_dir=reports)


def _int_list(raw: Any, field: str) -> List[int]:
    """Parse list of integers from YAML."""
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError(f"Field '{field}' must be a list of integers")
    values: List[int] = []
    for item in raw:
        try:
            values.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value in '{field}': {item}") from exc
    return values


def _percentiles_by_forw(raw: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """
    Parse mapping:
      {
        "1": {"alpha_pct": 0.30, "beta_pct": 0.90},
        "2": {"alpha_pct": 0.30, "beta_pct": 0.92},
        ...
      }
    into {1: {"alpha_pct": 0.30, "beta_pct": 0.90}, ...}
    """
    result: Dict[int, Dict[str, float]] = {}
    for key, payload in raw.items():
        try:
            window = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid forward window key: {key}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Percentile entry for forW={key} must be a mapping")
        try:
            alpha_pct = float(payload["alpha_pct"])
            beta_pct = float(payload["beta_pct"])
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing '{missing}' for percentile forW={window}") from exc
        result[window] = {"alpha_pct": alpha_pct, "beta_pct": beta_pct}
    return result


def _single_candle_percentiles(raw: Any) -> Dict[str, float]:
    """
    Parse single-candle percentiles mapping:
      {"alpha_pct": 0.25, "beta_pct": 0.997}
    Returns default values if not provided.
    """
    if raw is None:
        return {"alpha_pct": 0.25, "beta_pct": 0.997}
    if not isinstance(raw, dict):
        raise ValueError("single_candle_percentiles must be a mapping")
    try:
        alpha_pct = float(raw["alpha_pct"])
        beta_pct = float(raw["beta_pct"])
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing '{missing}' in single_candle_percentiles") from exc
    return {"alpha_pct": alpha_pct, "beta_pct": beta_pct}


def build_stage1_config(label_cfg: Path, paths_cfg: Path) -> Stage1Config:
    """
    Build Stage1Config supporting two calibration modes:
      - exact_paper: global alpha/beta from 1-candle returns (or constants),
                     beta scaled by +scale_beta_per_step per +1 in forW
      - per_forw_percentile: legacy/adaptive percentiles from |R_{t,forW}|
    """
    labeling_raw = read_yaml(label_cfg)
    paths_raw = read_yaml(paths_cfg)

    # Mandatory fields
    try:
        symbol = labeling_raw["symbol"]
        interval = labeling_raw["interval"]
        fee = float(labeling_raw["fee"])
        backW = _int_list(labeling_raw["backW"], "backW")
        forW = _int_list(labeling_raw["forW"], "forW")
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing '{missing}' entry in labeling config") from exc

    # Calibration mode either "exact_paper" (default) or "per_forw_percentile"
    calibration_mode = str(labeling_raw.get("calibration_mode", "exact_paper")).strip()

    # Paper-style extras
    alpha_const = labeling_raw.get("alpha_const", None)
    beta_const = labeling_raw.get("beta_const", None)
    alpha_const = float(alpha_const) if alpha_const is not None else None
    beta_const = float(beta_const) if beta_const is not None else None

    single_pct = _single_candle_percentiles(labeling_raw.get("single_candle_percentiles"))
    scale_beta_per_step = float(labeling_raw.get("scale_beta_per_step", 0.10))

    # Legacy per-forW percentiles (only required in that mode)
    percentiles: Dict[int, Dict[str, float]] = {}
    if calibration_mode == "per_forw_percentile":
        try:
            percentiles = _percentiles_by_forw(labeling_raw["percentiles_by_forW"])
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Missing '{missing}' entry in labeling config (required in per_forw_percentile mode)"
            ) from exc

    paths = build_paths_config(paths_raw)
    paths.ensure()

    return Stage1Config(
        symbol=symbol,
        interval=interval,
        fee=fee,
        backW_values=backW,
        forW_values=forW,
        percentiles_by_forW=percentiles,
        paths=paths,
        calibration_mode=calibration_mode,                  
        alpha_const=alpha_const,                            
        beta_const=beta_const,                              
        single_candle_percentiles=single_pct,               
        scale_beta_per_step=scale_beta_per_step,             # (+10% per step by default)
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """CLI arguments for Stage 1 pipeline."""
    parser = argparse.ArgumentParser(description="Run Stage 1 labeling pipeline")
    parser.add_argument(
        "--labeling-config",
        default="configs/labeling_paper.yaml",
        help="Path to labeling YAML configuration",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.yaml",
        help="Path to paths YAML configuration",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Entrypoint: load config -> run Stage 1 -> print QA summary."""
    args = parse_args(argv)
    label_cfg = Path(args.labeling_config)
    paths_cfg = Path(args.paths_config)

    try:
        cfg = build_stage1_config(label_cfg, paths_cfg)
    except Exception as exc:
        print(f"Failed to load configuration: {exc}")
        return 1

    pipeline = Stage1Pipeline(cfg)
    try:
        qa_report = pipeline.run()
    except Exception as exc:
        print(f"Stage 1 pipeline failed: {exc}")
        return 1

    print(
        "Stage 1 completed:",
        f"status={qa_report.status}",
        f"gaps_pct={qa_report.gaps_pct:.3f}",
        f"dup={qa_report.n_dupes}",
        f"nan={qa_report.n_nan}",
    )
    if qa_report.notes:
        print(f"Notes: {qa_report.notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())