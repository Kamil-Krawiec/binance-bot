from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sample_for_analysis(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=random_state)


def _target_series(df: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "action_vs_hold":
        return (df["label_id"] != 0).astype("int64")
    if strategy == "buy_vs_rest":
        return (df["label_id"] == 1).astype("int64")
    if strategy == "sell_vs_rest":
        return (df["label_id"] == 2).astype("int64")
    raise ValueError(f"Unsupported feature-analysis target strategy: {strategy}")


def run_xgboost_feature_importance(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    output_dir: Path,
    sample_size: int,
    top_n: int,
    random_state: int,
    target_strategy: str,
) -> dict[str, Any]:
    """Run the optional Data-explorer-library XGBoost feature importance analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from analysis.feature_importance import feature_importance
    except Exception as exc:
        return {"status": "skipped", "reason": f"missing_analysis_dependency:{exc}"}

    analysis_df = df[feature_columns + ["label_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    analysis_df = _sample_for_analysis(analysis_df, sample_size, random_state)
    y = _target_series(analysis_df, target_strategy)
    if y.nunique() < 2:
        return {"status": "skipped", "reason": "target_has_one_class"}

    scores = feature_importance(
        analysis_df[feature_columns],
        y,
        num_random_features=5,
        num_models=3,
        random_state=random_state,
        top_n=top_n,
        verbose=False,
    )
    scores = scores[~scores.index.str.startswith("rand_feat_")]
    scores_df = scores.rename("importance_score").reset_index().rename(columns={"index": "feature"})
    scores_path = output_dir / "xgboost_feature_importance_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    top_scores = scores_df.head(top_n).sort_values("importance_score")
    ax.barh(top_scores["feature"], top_scores["importance_score"], color="#2f6f73")
    ax.set_title(f"Top {top_n} Features: {target_strategy}")
    ax.set_xlabel("Aggregated importance score")
    fig.tight_layout()
    chart_path = output_dir / "xgboost_feature_importance_top.png"
    fig.savefig(chart_path, dpi=150)
    plt.close("all")

    return {
        "status": "completed",
        "target_strategy": target_strategy,
        "sample_rows": int(len(analysis_df)),
        "scores_path": str(scores_path),
        "chart_path": str(chart_path),
    }
