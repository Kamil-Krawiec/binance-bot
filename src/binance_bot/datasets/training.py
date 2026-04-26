from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from binance_bot.analysis.feature_analysis import run_xgboost_feature_importance
from binance_bot.config import TrainingDatasetConfig
from binance_bot.features.advanced import add_advanced_features
from binance_bot.features.registry import NON_MODEL_COLUMNS, columns_for_groups


@dataclass(frozen=True)
class TrainingDatasetResult:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    chart_paths: list[Path]
    train_rows: int
    validation_rows: int
    feature_columns: list[str]


def _label_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).sort_index().items()}


def _class_weights(df: pd.DataFrame) -> dict[str, float]:
    counts = df["label_id"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    if total == 0 or n_classes == 0:
        return {}
    return {str(label_id): float(total / (n_classes * count)) for label_id, count in sorted(counts.items())}


def _validation_end_timestamp(value: str | None, max_ts: pd.Timestamp) -> pd.Timestamp:
    if value is None or value.lower() == "today":
        today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return min(today, max_ts)
    return pd.to_datetime(value, utc=True)


def _effective_cutoff(config: TrainingDatasetConfig, dataset: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, str | None]:
    requested_cutoff = pd.to_datetime(config.split.validation_cutoff, utc=True)
    max_ts = dataset["ts"].max()
    if requested_cutoff <= max_ts:
        return requested_cutoff, requested_cutoff, None
    if config.split.fallback_last_available_months:
        effective = max_ts - pd.DateOffset(months=config.split.fallback_last_available_months)
        reason = (
            f"requested_cutoff_after_master_end:{requested_cutoff.isoformat()} > "
            f"{max_ts.isoformat()}; using last {config.split.fallback_last_available_months} available months"
        )
        return requested_cutoff, effective, reason
    return requested_cutoff, requested_cutoff, None


def _drop_feature_na(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("No feature columns selected")
    result = df.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=feature_columns)


def _lag_features_by_symbol(df: pd.DataFrame, feature_columns: list[str], lag_periods: int) -> pd.DataFrame:
    if lag_periods < 0:
        raise ValueError("feature lag_periods must be >= 0")
    if lag_periods == 0:
        return df
    result = df.sort_values(["symbol", "ts"]).copy()
    result[feature_columns] = result.groupby("symbol", sort=False)[feature_columns].shift(lag_periods)
    return result.sort_values(["ts", "symbol"]).reset_index(drop=True)


def _balance_train(df: pd.DataFrame, *, mode: str, hold_multiplier: float, random_state: int) -> pd.DataFrame:
    if mode == "none" or mode == "class_weights_only":
        return df.sort_values(["ts", "symbol"]).reset_index(drop=True)

    groups = {label_id: group for label_id, group in df.groupby("label_id")}
    if mode == "undersample_hold":
        hold = groups.get(0)
        non_hold = [group for label_id, group in groups.items() if label_id != 0]
        if hold is None or not non_hold:
            return df.sort_values(["ts", "symbol"]).reset_index(drop=True)
        max_action = max(len(group) for group in non_hold)
        hold_target = min(len(hold), int(max_action * hold_multiplier))
        balanced = [hold.sample(hold_target, random_state=random_state), *non_hold]
        return pd.concat(balanced, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)

    if mode == "equal_classes":
        if not groups:
            return df
        target = min(len(group) for group in groups.values())
        balanced = [
            group.sample(target, random_state=random_state) if len(group) > target else group
            for group in groups.values()
        ]
        return pd.concat(balanced, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)

    raise ValueError(f"Unsupported balance mode: {mode}")


def _write_charts(
    output_dir: Path,
    *,
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []

    label_counts = pd.DataFrame(
        {
            "train_before_balance": train_before["label"].value_counts(),
            "train_after_balance": train_after["label"].value_counts(),
            "validation": validation["label"].value_counts(),
        }
    ).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("Label Distribution")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    path = charts_dir / "label_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    chart_paths.append(path)

    split_counts = pd.Series(
        {
            "train_before_balance": len(train_before),
            "train_after_balance": len(train_after),
            "validation": len(validation),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    split_counts.plot(kind="bar", ax=ax, color="#4361ee")
    ax.set_title("Rows By Split")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = charts_dir / "rows_by_split.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    chart_paths.append(path)

    by_month = pd.concat(
        [
            train_before.assign(split="train_before_balance"),
            train_after.assign(split="train_after_balance"),
            validation.assign(split="validation"),
        ],
        ignore_index=True,
    )
    by_month["month"] = by_month["ts"].dt.strftime("%Y-%m")
    monthly = by_month.groupby(["month", "split", "label"]).size().reset_index(name="rows")
    for split_name in ["train_before_balance", "train_after_balance", "validation"]:
        subset = monthly[monthly["split"] == split_name]
        if subset.empty:
            continue
        pivot = subset.pivot(index="month", columns="label", values="rows").fillna(0)
        fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.25), 5))
        pivot.plot(kind="area", stacked=True, ax=ax, color=["#2ca02c", "#7f7f7f", "#d62728"])
        ax.set_title(f"Monthly Label Counts: {split_name}")
        ax.set_ylabel("Rows")
        ax.set_xlabel("Month")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        path = charts_dir / f"monthly_label_counts_{split_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        chart_paths.append(path)

    validation_symbols = validation.groupby("symbol").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    validation_symbols.head(40).sort_values().plot(kind="barh", ax=ax, color="#7b2cbf")
    ax.set_title("Validation Rows By Symbol: Top 40")
    ax.set_xlabel("Rows")
    fig.tight_layout()
    path = charts_dir / "validation_rows_by_symbol_top40.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    chart_paths.append(path)

    numeric = train_after[feature_columns + ["label_id"]].select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)["label_id"].drop("label_id", errors="ignore").abs().sort_values(ascending=False)
    corr.head(30).rename("abs_corr_with_label_id").to_csv(charts_dir / "feature_target_abs_correlation.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    top_corr = corr.head(30).sort_values()
    ax.barh(top_corr.index, top_corr.values, color="#00876c")
    ax.set_title("Top Absolute Correlations With label_id")
    ax.set_xlabel("Absolute Pearson correlation")
    fig.tight_layout()
    path = charts_dir / "feature_target_abs_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    chart_paths.append(path)

    return chart_paths


def prepare_training_dataset(config: TrainingDatasetConfig) -> TrainingDatasetResult:
    if not config.master_path.exists():
        raise FileNotFoundError(config.master_path)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(config.master_path)
    master["ts"] = pd.to_datetime(master["ts"], utc=True)
    master = master.sort_values(["symbol", "ts"]).reset_index(drop=True)
    enriched = add_advanced_features(master)

    selected = columns_for_groups(config.feature_selection.include_groups, enriched.columns)
    leakage_excluded = set(config.feature_selection.leakage_exclude_columns)
    excluded = set(config.feature_selection.exclude_columns) | leakage_excluded | NON_MODEL_COLUMNS
    feature_columns = [
        column
        for column in selected
        if column not in excluded and pd.api.types.is_numeric_dtype(enriched[column])
    ]
    dropped_non_numeric = [column for column in selected if column not in excluded and column not in feature_columns]

    model_columns = ["ts", "symbol", "label", "label_id", *feature_columns]
    dataset = _lag_features_by_symbol(
        enriched[model_columns],
        feature_columns,
        config.feature_selection.lag_periods,
    )
    dataset = _drop_feature_na(dataset, feature_columns)

    requested_cutoff, cutoff, cutoff_fallback_reason = _effective_cutoff(config, dataset)
    validation_end = _validation_end_timestamp(config.split.validation_end, dataset["ts"].max())
    train_before_balance = dataset[dataset["ts"] < cutoff].copy()
    validation = dataset[(dataset["ts"] >= cutoff) & (dataset["ts"] <= validation_end)].copy()
    if train_before_balance.empty:
        raise RuntimeError(f"Training split is empty before cutoff {cutoff}")
    if validation.empty:
        raise RuntimeError(f"Validation split is empty from {cutoff} to {validation_end}")

    train_after_balance = _balance_train(
        train_before_balance,
        mode=config.balance.mode,
        hold_multiplier=config.balance.hold_multiplier,
        random_state=config.balance.random_state,
    )

    stem = "b5_f2"
    if {"backW", "forW"}.issubset(master.columns):
        stem = f"b{int(master['backW'].mode().iloc[0])}_f{int(master['forW'].mode().iloc[0])}"
    train_path = output_dir / f"train_{stem}.parquet"
    validation_path = output_dir / f"validation_{stem}.parquet"
    train_after_balance.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)

    chart_paths = _write_charts(
        output_dir,
        train_before=train_before_balance,
        train_after=train_after_balance,
        validation=validation,
        feature_columns=feature_columns,
    )

    analysis_dir = output_dir / "analysis"
    feature_analysis: dict[str, Any] = {"status": "disabled"}
    if config.analysis.enabled:
        feature_analysis = run_xgboost_feature_importance(
            train_after_balance,
            feature_columns,
            output_dir=analysis_dir,
            sample_size=config.analysis.sample_size,
            top_n=config.analysis.top_n,
            random_state=config.analysis.random_state,
            target_strategy=config.analysis.target_strategy,
        )

    manifest = {
        "source_master": str(config.master_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "requested_cutoff": requested_cutoff.isoformat(),
        "effective_cutoff": cutoff.isoformat(),
        "cutoff_fallback_reason": cutoff_fallback_reason,
        "validation_end": validation_end.isoformat(),
        "feature_groups": list(config.feature_selection.include_groups),
        "feature_columns": feature_columns,
        "excluded_columns": sorted(excluded),
        "leakage_excluded_columns": sorted(leakage_excluded),
        "feature_lag_periods": config.feature_selection.lag_periods,
        "dropped_non_numeric_selected_columns": dropped_non_numeric,
        "balance": {
            "mode": config.balance.mode,
            "hold_multiplier": config.balance.hold_multiplier,
            "random_state": config.balance.random_state,
        },
        "rows": {
            "source_after_feature_na_drop": int(len(dataset)),
            "train_before_balance": int(len(train_before_balance)),
            "train_after_balance": int(len(train_after_balance)),
            "validation": int(len(validation)),
        },
        "symbols": {
            "train": int(train_after_balance["symbol"].nunique()),
            "validation": int(validation["symbol"].nunique()),
        },
        "label_distribution": {
            "train_before_balance": _label_distribution(train_before_balance),
            "train_after_balance": _label_distribution(train_after_balance),
            "validation": _label_distribution(validation),
        },
        "class_weights_train_before_balance": _class_weights(train_before_balance),
        "charts": [str(path) for path in chart_paths],
        "feature_analysis": feature_analysis,
    }

    manifest_path = output_dir / f"feature_manifest_{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return TrainingDatasetResult(
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        chart_paths=chart_paths,
        train_rows=len(train_after_balance),
        validation_rows=len(validation),
        feature_columns=feature_columns,
    )
