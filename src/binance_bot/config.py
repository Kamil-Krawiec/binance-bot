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
