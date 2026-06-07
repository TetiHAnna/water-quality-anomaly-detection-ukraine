"""
01_model_validation.py

Model validation script for surface-water quality anomaly detection study.

This script reproduces the 5-fold cross-validation results reported in Table 2.
It evaluates Random Forest and XGBoost regression models for two target variables:
    - Azot  : nitrogen concentration
    - Kisen : dissolved oxygen

Input file expected in the working directory:
    combined_water_quality.csv

Output file:
    cv_results_table2.csv

Computational complexity:
    Let n be the number of records after target-specific cleaning, p the number
    of predictors, k the number of CV folds, T the number of trees, and d the
    average tree depth.

    Random Forest training is approximately O(k * T * n * p * log n).
    XGBoost training is approximately O(k * T * n * p * d), depending on the
    tree construction method used internally by XGBoost.

    Memory complexity is approximately O(n * p) for the feature matrix plus the
    memory required to store the ensemble trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor


DATA_PATH = Path("combined_water_quality.csv")
OUTPUT_PATH = Path("cv_results_table2.csv")
RANDOM_STATE = 42
N_SPLITS = 5

FEATURES = [
    "BSK5",
    "Fosfat",
    "Nitrat",
    "month",
    "Latitude",
    "Longitude",
]


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for a target variable used in model validation."""

    target_name: str
    indicator_label: str


TARGETS = [
    TargetConfig(target_name="Azot", indicator_label="Nitrogen / Azot"),
    TargetConfig(target_name="Kisen", indicator_label="Dissolved oxygen / Kisen"),
]


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean square error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the source dataset and validate that required columns are present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Place 'combined_water_quality.csv' "
            "in the working directory and run the script again."
        )

    df = pd.read_csv(path)
    required_columns = set(FEATURES + ["Azot", "Kisen", "Post_Name"])
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "The input dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def prepare_target_data(
    df: pd.DataFrame,
    target_name: str,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """
    Remove missing values for a specific target and return X, y, n, and sites.

    Missing-value removal is performed separately for each target variable. This
    is intentional because nitrogen and dissolved oxygen have different data
    completeness patterns in the source monitoring dataset.
    """
    cleaned = df.dropna(subset=[target_name] + FEATURES + ["Post_Name"]).copy()

    X = cleaned[FEATURES]
    y = cleaned[target_name]
    n_records = int(len(cleaned))
    n_sites = int(cleaned["Post_Name"].nunique())

    if n_records == 0:
        raise ValueError(f"No records remain after cleaning for target: {target_name}")

    return X, y, n_records, n_sites


def build_models() -> dict[str, object]:
    """Create regression models with reproducible hyperparameter settings."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=1,
        ),
    }


def format_mean_sd(values: np.ndarray) -> str:
    """Format mean ± standard deviation using three decimal places."""
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def evaluate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[str, str]:
    """Evaluate one model using 5-fold cross-validation."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "r2": "r2",
        "rmse": make_scorer(rmse_score, greater_is_better=False),
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    r2_values = scores["test_r2"]
    rmse_values = -scores["test_rmse"]

    return format_mean_sd(r2_values), format_mean_sd(rmse_values)


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the cross-validation results table for all targets and models."""
    rows: list[dict[str, object]] = []
    models = build_models()

    for target in TARGETS:
        X, y, n_records, n_sites = prepare_target_data(df, target.target_name)

        for model_name, model in models.items():
            r2_text, rmse_text = evaluate_model(model, X, y)
            rows.append(
                {
                    "Indicator": target.indicator_label,
                    "Model": model_name,
                    "n": n_records,
                    "Sites": n_sites,
                    "R², mean ± SD": r2_text,
                    "RMSE, mean ± SD": rmse_text,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run model validation and save Table 2 cross-validation results."""
    df = load_dataset(DATA_PATH)
    results = build_results_table(df)

    print("Table 2. Five-fold cross-validation results")
    print(results.to_string(index=False))

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
