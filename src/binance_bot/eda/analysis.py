from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from binance_bot.config import EDAConfig


@dataclass(frozen=True)
class EDAResult:
    output_dir: Path
    overview_path: Path
    table_paths: list[Path]
    chart_paths: list[Path]


def _safe_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _ensure_numeric(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _dataset_overview(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: EDAConfig,
    feature_columns: list[str],
) -> dict[str, Any]:
    def split_payload(df: pd.DataFrame) -> dict[str, Any]:
        numeric = df[feature_columns].select_dtypes(include="number")
        return {
            "rows": int(len(df)),
            "date_min": df[config.time_column].min(),
            "date_max": df[config.time_column].max(),
            "symbols": int(df[config.symbol_column].nunique()),
            "label_distribution": {
                str(label): int(count)
                for label, count in df[config.label_column].value_counts(dropna=False).sort_index().items()
            },
            "missing_numeric_values": int(numeric.isna().sum().sum()),
            "infinite_numeric_values": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
        }

    return {
        "train": split_payload(train),
        "validation": split_payload(validation),
        "feature_count": len(feature_columns),
        "target_column": config.target_column,
        "label_column": config.label_column,
    }


def _feature_quality(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    n = len(df)
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        q1 = finite.quantile(0.25) if not finite.empty else np.nan
        q3 = finite.quantile(0.75) if not finite.empty else np.nan
        iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
        if pd.notna(iqr) and iqr > 0:
            outlier_pct = float(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).mean() * 100)
        else:
            outlier_pct = 0.0
        records.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "missing_pct": float(values.isna().mean() * 100),
                "inf_pct": float(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).mean() * 100),
                "unique_count": int(values.nunique(dropna=True)),
                "unique_pct": float(values.nunique(dropna=True) / n * 100) if n else 0.0,
                "mean": float(finite.mean()) if not finite.empty else np.nan,
                "std": float(finite.std()) if not finite.empty else np.nan,
                "skew": float(finite.skew()) if len(finite) > 2 else np.nan,
                "kurtosis": float(finite.kurtosis()) if len(finite) > 3 else np.nan,
                "iqr_outlier_pct": outlier_pct,
                "zero_variance": bool(finite.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(records)


def _target_influence(
    train: pd.DataFrame,
    config: EDAConfig,
    feature_columns: list[str],
    xgb_importance: pd.DataFrame | None,
) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif

    numeric = _ensure_numeric(train, feature_columns)
    target = train[config.target_column].astype(int)
    corr = numeric.corrwith(target).rename("pearson_corr_label_id")
    influence = corr.to_frame()
    influence["abs_corr_label_id"] = influence["pearson_corr_label_id"].abs()

    for label_name, label_id in {"BUY": 1, "HOLD": 0, "SELL": 2}.items():
        one_vs_rest = (target == label_id).astype(int)
        influence[f"corr_{label_name.lower()}_vs_rest"] = numeric.corrwith(one_vs_rest)
        influence[f"abs_corr_{label_name.lower()}_vs_rest"] = influence[f"corr_{label_name.lower()}_vs_rest"].abs()

    mi_frame = pd.concat([numeric, target.rename(config.target_column)], axis=1).dropna()
    mi_sample = _safe_sample(mi_frame, config.sample.mutual_info_rows, config.sample.random_state)
    if not mi_sample.empty and mi_sample[config.target_column].nunique() > 1:
        mi_values = mutual_info_classif(
            mi_sample[feature_columns],
            mi_sample[config.target_column].astype(int),
            random_state=config.sample.random_state,
            discrete_features=False,
        )
        influence["mutual_info_label_id"] = pd.Series(mi_values, index=feature_columns)
    else:
        influence["mutual_info_label_id"] = np.nan

    if xgb_importance is not None and not xgb_importance.empty:
        importance = xgb_importance.set_index("feature")["importance_score"]
        influence["xgboost_importance_score"] = importance
    else:
        influence["xgboost_importance_score"] = np.nan

    influence = influence.reset_index().rename(columns={"index": "feature"})
    sort_cols = ["xgboost_importance_score", "mutual_info_label_id", "abs_corr_label_id"]
    return influence.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)


def _drift(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp
    except Exception:  # pragma: no cover
        ks_2samp = None

    rows: list[dict[str, Any]] = []
    train_numeric = _ensure_numeric(train, feature_columns)
    validation_numeric = _ensure_numeric(validation, feature_columns)
    for column in feature_columns:
        train_values = train_numeric[column].dropna()
        validation_values = validation_numeric[column].dropna()
        train_sample = _safe_sample(train_values.to_frame(column), 50_000, config.sample.random_state)[column]
        validation_sample = _safe_sample(validation_values.to_frame(column), 50_000, config.sample.random_state)[column]
        if ks_2samp is not None and len(train_sample) > 0 and len(validation_sample) > 0:
            ks = ks_2samp(train_sample, validation_sample)
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        else:
            ks_stat = np.nan
            ks_pvalue = np.nan
        train_mean = float(train_values.mean()) if len(train_values) else np.nan
        validation_mean = float(validation_values.mean()) if len(validation_values) else np.nan
        train_std = float(train_values.std()) if len(train_values) else np.nan
        validation_std = float(validation_values.std()) if len(validation_values) else np.nan
        std_ratio = validation_std / train_std if pd.notna(train_std) and train_std != 0 else np.nan
        rows.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "validation_mean": validation_mean,
                "mean_diff": validation_mean - train_mean,
                "train_std": train_std,
                "validation_std": validation_std,
                "std_ratio_validation_train": std_ratio,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False, na_position="last")


def _per_symbol_labels(df: pd.DataFrame, config: EDAConfig, split: str) -> pd.DataFrame:
    counts = (
        df.groupby([config.symbol_column, config.label_column])
        .size()
        .rename("rows")
        .reset_index()
        .pivot(index=config.symbol_column, columns=config.label_column, values="rows")
        .fillna(0)
        .astype(int)
    )
    counts["total_rows"] = counts.sum(axis=1)
    for label in ["BUY", "HOLD", "SELL"]:
        if label not in counts.columns:
            counts[label] = 0
        counts[f"{label.lower()}_pct"] = counts[label] / counts["total_rows"] * 100
    counts["split"] = split
    return counts.reset_index().sort_values("total_rows", ascending=False)


def _high_correlation_pairs(df: pd.DataFrame, feature_columns: list[str], threshold: float) -> pd.DataFrame:
    numeric = _ensure_numeric(df, feature_columns).dropna()
    corr = numeric.corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)


def _normalization_recommendations(quality: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in quality.iterrows():
        feature = str(row["feature"])
        unique_count = int(row["unique_count"])
        skew = abs(float(row["skew"])) if pd.notna(row["skew"]) else 0.0
        kurtosis = float(row["kurtosis"]) if pd.notna(row["kurtosis"]) else 0.0
        outlier_pct = float(row["iqr_outlier_pct"])
        is_binary = unique_count <= 2 or feature.startswith("fng_is_") or feature.startswith("fng_extreme_")
        is_time = feature in {"month", "day_of_week", "hour"}
        heavy_tail = skew > 2.0 or kurtosis > 10.0 or outlier_pct > 5.0
        if is_binary:
            mlp = "passthrough_binary"
            reason = "binary_or_flag"
        elif is_time:
            mlp = "passthrough_or_cyclical_encoding"
            reason = "calendar_feature"
        elif heavy_tail:
            mlp = "RobustScaler_candidate"
            reason = "heavy_tailed_or_many_outliers"
        else:
            mlp = "StandardScaler"
            reason = "continuous_numeric"
        rows.append(
            {
                "feature": feature,
                "tree_models": "no_scaling_required",
                "mlp_logistic": mlp,
                "reason": reason,
                "skew_abs": skew,
                "kurtosis": kurtosis,
                "iqr_outlier_pct": outlier_pct,
                "unique_count": unique_count,
            }
        )
    return pd.DataFrame(rows)


def _write_basic_charts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    influence: pd.DataFrame,
    drift: pd.DataFrame,
    per_symbol_validation: pd.DataFrame,
    config: EDAConfig,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    label_counts = pd.DataFrame(
        {
            "train": train[config.label_column].value_counts(),
            "validation": validation[config.label_column].value_counts(),
        }
    ).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("Target Distribution By Split")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    path = charts_dir / "target_distribution_by_split.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    for column, filename, title in [
        ("abs_corr_label_id", "top_abs_correlation.png", "Top Absolute Correlation With label_id"),
        ("mutual_info_label_id", "top_mutual_information.png", "Top Mutual Information With label_id"),
        ("xgboost_importance_score", "top_xgboost_importance.png", "Top XGBoost Importance"),
    ]:
        subset = influence.dropna(subset=[column]).head(config.analysis.top_n).sort_values(column)
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(5, len(subset) * 0.32)))
        ax.barh(subset["feature"], subset[column], color="#2f6f73")
        ax.set_title(title)
        ax.set_xlabel(column)
        fig.tight_layout()
        path = charts_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    drift_subset = drift.head(config.analysis.top_drift_features).sort_values("ks_statistic")
    if not drift_subset.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, len(drift_subset) * 0.32)))
        ax.barh(drift_subset["feature"], drift_subset["ks_statistic"], color="#b56576")
        ax.set_title("Top Train vs Validation Drift By KS Statistic")
        ax.set_xlabel("KS statistic")
        fig.tight_layout()
        path = charts_dir / "top_feature_drift_ks.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    symbol_subset = per_symbol_validation.head(40).sort_values("total_rows")
    if not symbol_subset.empty:
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(symbol_subset[config.symbol_column], symbol_subset["total_rows"], color="#7b2cbf")
        ax.set_title("Validation Rows By Symbol: Top 40")
        ax.set_xlabel("Rows")
        fig.tight_layout()
        path = charts_dir / "validation_rows_by_symbol_top40.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        pct_columns = ["buy_pct", "hold_pct", "sell_pct"]
        fig, ax = plt.subplots(figsize=(10, 9))
        left = np.zeros(len(symbol_subset))
        colors = {"buy_pct": "#2f6f73", "hold_pct": "#8d99ae", "sell_pct": "#b56576"}
        labels = {"buy_pct": "BUY", "hold_pct": "HOLD", "sell_pct": "SELL"}
        for column in pct_columns:
            values = symbol_subset[column].to_numpy()
            ax.barh(
                symbol_subset[config.symbol_column],
                values,
                left=left,
                color=colors[column],
                label=labels[column],
            )
            left += values
        ax.set_title("Validation Label Distribution By Symbol: Top 40")
        ax.set_xlabel("Percent of symbol rows")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = charts_dir / "validation_label_distribution_by_symbol_top40.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    top_corr_features = influence["feature"].head(min(20, len(influence))).tolist()
    if top_corr_features:
        corr = train[top_corr_features].corr().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Heatmap: Top Influence Features")
        fig.tight_layout()
        path = charts_dir / "top_feature_correlation_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    top_features = influence["feature"].head(config.analysis.top_distribution_features).tolist()
    plot_df = _safe_sample(
        pd.concat([train.assign(split="train"), validation.assign(split="validation")], ignore_index=True),
        config.sample.plot_rows,
        config.sample.random_state,
    )
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, group in plot_df.groupby(config.label_column):
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) == 0:
                continue
            ax.hist(values, bins=60, alpha=0.35, density=True, label=str(label))
        ax.set_title(f"{feature} Distribution By {config.label_column}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        path = charts_dir / f"distribution_by_target_{feature}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    for feature in drift["feature"].head(min(8, len(drift))).tolist():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(
            [
                train[feature].replace([np.inf, -np.inf], np.nan).dropna(),
                validation[feature].replace([np.inf, -np.inf], np.nan).dropna(),
            ],
            tick_labels=["train", "validation"],
            showfliers=False,
        )
        ax.set_title(f"Train vs Validation Distribution: {feature}")
        ax.set_ylabel(feature)
        fig.tight_layout()
        path = charts_dir / f"drift_boxplot_{feature}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _run_library_plots(
    train: pd.DataFrame,
    influence: pd.DataFrame,
    config: EDAConfig,
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    if not config.analysis.library_enabled:
        return [], {"status": "disabled"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    library_dir = output_dir / "library_plots"
    library_dir.mkdir(parents=True, exist_ok=True)
    sample = _safe_sample(train, config.sample.library_rows, config.sample.random_state)
    selected = influence["feature"].head(min(6, len(influence))).tolist()
    paths: list[Path] = []
    status: dict[str, str] = {"status": "completed"}

    try:
        from pre_analysis.category_proportion import category_proportion_bar

        path = library_dir / "label_proportion_bar.png"
        category_proportion_bar(sample, config.label_column, title="Label Proportion", filepath=str(path), add_labels=True)
        plt.close("all")
        paths.append(path)

        path = library_dir / "symbol_proportion_bar.png"
        top_symbols = sample[config.symbol_column].value_counts().head(30).index
        category_proportion_bar(
            sample[sample[config.symbol_column].isin(top_symbols)],
            config.symbol_column,
            title="Symbol Proportion: Top 30 In Sample",
            filepath=str(path),
            add_labels=False,
        )
        plt.close("all")
        paths.append(path)
    except Exception as exc:
        status["category_proportion_bar"] = str(exc)

    try:
        from pre_analysis.feature_distribution_by_target import feature_distribution_by_target

        for feature in selected[:4]:
            feature_distribution_by_target(sample, feature, config.label_column, directory=str(library_dir))
            path = library_dir / f"{feature}_by_{config.label_column}.png"
            if path.exists():
                paths.append(path)
            plt.close("all")
    except Exception as exc:
        status["feature_distribution_by_target"] = str(exc)

    try:
        from pre_analysis.distribution import visualize_distribution

        visualize_distribution(sample, selected[:3], save_to_file=str(library_dir))
        for feature in selected[:3]:
            path = library_dir / f"visualize_distribution_{feature}.png"
            if path.exists():
                paths.append(path)
        plt.close("all")
    except Exception as exc:
        status["visualize_distribution"] = str(exc)

    try:
        from pre_analysis.outliers import detect_outliers_3d

        outlier_features = [feature for feature in selected if feature in sample.columns][:3]
        if len(outlier_features) == 3:
            outliers, fig = detect_outliers_3d(
                sample[outlier_features].dropna(),
                n_outliers=50,
                detection_features=outlier_features,
                visualization_features=outlier_features,
                random_state=config.sample.random_state,
            )
            outliers.to_csv(library_dir / "outliers_3d_sample.csv", index=False)
            path = library_dir / "outliers_3d.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    except Exception as exc:
        status["detect_outliers_3d"] = str(exc)

    try:
        from analysis.feature_importance import feature_correlation_with_target

        corr_df = feature_correlation_with_target(
            sample[selected[:4] + [config.target_column]].dropna(),
            selected[:4],
            config.target_column,
            save_to_file=str(library_dir),
        )
        corr_df.to_csv(library_dir / "library_feature_correlation_with_target.csv", index=False)
        paths.append(library_dir / "summary_correlation_plot.png")
        plt.close("all")
    except Exception as exc:
        status["feature_correlation_with_target"] = str(exc)

    return paths, status


def run_eda_analysis(config: EDAConfig) -> EDAResult:
    output_dir = config.output_dir
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(config.train_path)
    validation = pd.read_parquet(config.validation_path)
    train[config.time_column] = pd.to_datetime(train[config.time_column], utc=True)
    validation[config.time_column] = pd.to_datetime(validation[config.time_column], utc=True)

    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    feature_columns = [feature for feature in manifest["feature_columns"] if feature in train.columns]

    xgb_importance = None
    if config.xgboost_importance_path and config.xgboost_importance_path.exists():
        xgb_importance = pd.read_csv(config.xgboost_importance_path)

    overview = _dataset_overview(train, validation, config, feature_columns)
    overview["source_files"] = {
        "train": str(config.train_path),
        "validation": str(config.validation_path),
        "manifest": str(config.manifest_path),
        "xgboost_importance": str(config.xgboost_importance_path) if config.xgboost_importance_path else None,
    }
    overview_path = output_dir / "eda_overview.json"
    _write_json(overview_path, overview)

    train_quality = _feature_quality(train, feature_columns)
    validation_quality = _feature_quality(validation, feature_columns)
    influence = _target_influence(train, config, feature_columns, xgb_importance)
    drift = _drift(train, validation, feature_columns, config)
    per_symbol = pd.concat(
        [
            _per_symbol_labels(train, config, "train"),
            _per_symbol_labels(validation, config, "validation"),
        ],
        ignore_index=True,
    )
    high_corr_pairs = _high_correlation_pairs(train, feature_columns, config.analysis.correlation_threshold)
    normalization = _normalization_recommendations(train_quality)

    table_payloads = {
        "feature_quality_train.csv": train_quality,
        "feature_quality_validation.csv": validation_quality,
        "target_influence.csv": influence,
        "feature_drift_train_vs_validation.csv": drift,
        "per_symbol_label_distribution.csv": per_symbol,
        "high_correlation_pairs.csv": high_corr_pairs,
        "normalization_recommendations.csv": normalization,
    }
    table_paths: list[Path] = []
    for filename, frame in table_payloads.items():
        path = tables_dir / filename
        frame.to_csv(path, index=False)
        table_paths.append(path)

    chart_paths = _write_basic_charts(train, validation, influence, drift, per_symbol[per_symbol["split"] == "validation"], config, output_dir)
    library_paths, library_status = _run_library_plots(train, influence, config, output_dir)
    chart_paths.extend(library_paths)
    _write_json(output_dir / "library_integration_status.json", library_status)

    return EDAResult(
        output_dir=output_dir,
        overview_path=overview_path,
        table_paths=table_paths,
        chart_paths=chart_paths,
    )
