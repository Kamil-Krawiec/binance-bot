from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FearGreedConfig:
    enabled: bool = True
    path: Path | None = None


@dataclass(frozen=True)
class FeatureConfig:
    technical: bool = True
    fear_greed: FearGreedConfig = FearGreedConfig()


@dataclass(frozen=True)
class MasterDatasetConfig:
    backW: int
    forW: int
    processed_dir: Path
    labeled_datasets_dir: Path
    symbols_file: Path | None
    output_path: Path
    features: FeatureConfig = FeatureConfig()


@dataclass(frozen=True)
class TrainingFeatureSelectionConfig:
    include_groups: tuple[str, ...]
    exclude_columns: tuple[str, ...] = ()
    leakage_exclude_columns: tuple[str, ...] = ()
    lag_periods: int = 1


@dataclass(frozen=True)
class TrainingSplitConfig:
    validation_cutoff: str
    validation_end: str | None = "today"
    fallback_last_available_months: int | None = None


@dataclass(frozen=True)
class BalanceConfig:
    mode: str = "undersample_hold"
    hold_multiplier: float = 1.5
    random_state: int = 42


@dataclass(frozen=True)
class TrainingAnalysisConfig:
    enabled: bool = True
    sample_size: int = 50_000
    top_n: int = 30
    random_state: int = 42
    target_strategy: str = "action_vs_hold"


@dataclass(frozen=True)
class TrainingDatasetConfig:
    master_path: Path
    output_dir: Path
    feature_selection: TrainingFeatureSelectionConfig
    split: TrainingSplitConfig
    balance: BalanceConfig = BalanceConfig()
    analysis: TrainingAnalysisConfig = TrainingAnalysisConfig()


@dataclass(frozen=True)
class EDASampleConfig:
    random_state: int = 42
    mutual_info_rows: int = 100_000
    plot_rows: int = 50_000
    library_rows: int = 20_000


@dataclass(frozen=True)
class EDAAnalysisConfig:
    top_n: int = 20
    top_distribution_features: int = 12
    top_drift_features: int = 15
    correlation_threshold: float = 0.95
    library_enabled: bool = True


@dataclass(frozen=True)
class EDAConfig:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    xgboost_importance_path: Path | None
    output_dir: Path
    target_column: str = "label_id"
    label_column: str = "label"
    symbol_column: str = "symbol"
    time_column: str = "ts"
    sample: EDASampleConfig = EDASampleConfig()
    analysis: EDAAnalysisConfig = EDAAnalysisConfig()


def _resolve_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_master_dataset_config(path: str | Path) -> MasterDatasetConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_root = config_path.resolve().parent.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(payload).__name__}")

    features_payload: dict[str, Any] = payload.get("features") or {}
    fear_payload: dict[str, Any] = features_payload.get("fear_greed") or {}

    fear_greed = FearGreedConfig(
        enabled=bool(fear_payload.get("enabled", True)),
        path=_resolve_path(project_root, fear_payload.get("path")),
    )

    features = FeatureConfig(
        technical=bool(features_payload.get("technical", True)),
        fear_greed=fear_greed,
    )

    return MasterDatasetConfig(
        backW=int(payload["backW"]),
        forW=int(payload["forW"]),
        processed_dir=_resolve_path(project_root, payload["processed_dir"]) or project_root,
        labeled_datasets_dir=_resolve_path(project_root, payload["labeled_datasets_dir"]) or project_root,
        symbols_file=_resolve_path(project_root, payload.get("symbols_file")),
        output_path=_resolve_path(project_root, payload["output_path"]) or project_root / "master.parquet",
        features=features,
    )


def load_training_dataset_config(path: str | Path) -> TrainingDatasetConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_root = config_path.resolve().parent.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(payload).__name__}")

    feature_payload: dict[str, Any] = payload.get("features") or {}
    split_payload: dict[str, Any] = payload.get("split") or {}
    balance_payload: dict[str, Any] = payload.get("balance") or {}
    analysis_payload: dict[str, Any] = payload.get("analysis") or {}

    include_groups = tuple(feature_payload.get("include_groups") or ["technical", "advanced", "time", "sentiment"])
    exclude_columns = tuple(feature_payload.get("exclude_columns") or [])
    leakage_exclude_columns = tuple(
        feature_payload.get("leakage_exclude_columns")
        or ["return_1", "log_return_1", "pct_change"]
    )
    lag_periods = int(feature_payload.get("lag_periods", 1))

    validation_cutoff = str(split_payload.get("validation_cutoff", "2025-01-01"))
    validation_end = split_payload.get("validation_end", "today")
    if validation_end is not None:
        validation_end = str(validation_end)
    fallback_months = split_payload.get("fallback_last_available_months")
    if fallback_months is not None:
        fallback_months = int(fallback_months)

    return TrainingDatasetConfig(
        master_path=_resolve_path(project_root, payload["master_path"]) or project_root,
        output_dir=_resolve_path(project_root, payload["output_dir"]) or project_root / "training",
        feature_selection=TrainingFeatureSelectionConfig(
            include_groups=include_groups,
            exclude_columns=exclude_columns,
            leakage_exclude_columns=leakage_exclude_columns,
            lag_periods=lag_periods,
        ),
        split=TrainingSplitConfig(
            validation_cutoff=validation_cutoff,
            validation_end=validation_end,
            fallback_last_available_months=fallback_months,
        ),
        balance=BalanceConfig(
            mode=str(balance_payload.get("mode", "undersample_hold")),
            hold_multiplier=float(balance_payload.get("hold_multiplier", 1.5)),
            random_state=int(balance_payload.get("random_state", 42)),
        ),
        analysis=TrainingAnalysisConfig(
            enabled=bool(analysis_payload.get("enabled", True)),
            sample_size=int(analysis_payload.get("sample_size", 50_000)),
            top_n=int(analysis_payload.get("top_n", 30)),
            random_state=int(analysis_payload.get("random_state", 42)),
            target_strategy=str(analysis_payload.get("target_strategy", "action_vs_hold")),
        ),
    )


def load_eda_config(path: str | Path) -> EDAConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_root = config_path.resolve().parent.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(payload).__name__}")

    sample_payload: dict[str, Any] = payload.get("sample") or {}
    analysis_payload: dict[str, Any] = payload.get("analysis") or {}

    return EDAConfig(
        train_path=_resolve_path(project_root, payload["train_path"]) or project_root,
        validation_path=_resolve_path(project_root, payload["validation_path"]) or project_root,
        manifest_path=_resolve_path(project_root, payload["manifest_path"]) or project_root,
        xgboost_importance_path=_resolve_path(project_root, payload.get("xgboost_importance_path")),
        output_dir=_resolve_path(project_root, payload["output_dir"]) or project_root / "eda",
        target_column=str(payload.get("target_column", "label_id")),
        label_column=str(payload.get("label_column", "label")),
        symbol_column=str(payload.get("symbol_column", "symbol")),
        time_column=str(payload.get("time_column", "ts")),
        sample=EDASampleConfig(
            random_state=int(sample_payload.get("random_state", 42)),
            mutual_info_rows=int(sample_payload.get("mutual_info_rows", 100_000)),
            plot_rows=int(sample_payload.get("plot_rows", 50_000)),
            library_rows=int(sample_payload.get("library_rows", 20_000)),
        ),
        analysis=EDAAnalysisConfig(
            top_n=int(analysis_payload.get("top_n", 20)),
            top_distribution_features=int(analysis_payload.get("top_distribution_features", 12)),
            top_drift_features=int(analysis_payload.get("top_drift_features", 15)),
            correlation_threshold=float(analysis_payload.get("correlation_threshold", 0.95)),
            library_enabled=bool(analysis_payload.get("library_enabled", True)),
        ),
    )
