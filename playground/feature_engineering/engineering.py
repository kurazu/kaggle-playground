from pathlib import Path
from typing import Dict, List, Optional, cast

import polars as pl

from ..utils import write_json
from .config import FeatureConfig
from .raw import scan_raw_dataset


def input_feature_engineering(
    raw_df: pl.LazyFrame, target_column: Optional[str]
) -> pl.LazyFrame:
    """
    Process raw input features into their feature engineered versions.
    """

    return raw_df.select(
        [
            pl.col("id"),
            pl.col("gender"),
            pl.col("age"),
            pl.when(pl.col("hypertension") == 1)
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias("hypertension"),
            pl.when(pl.col("heart_disease") == 1)
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias("heart_disease"),
            pl.col("ever_married"),
            pl.col("work_type"),
            pl.col("Residence_type").alias("residence_type"),
            pl.col("avg_glucose_level"),
            pl.col("bmi"),
            pl.col("smoking_status"),
        ]
        + (
            [pl.col(target_column).alias("classification_target")]
            if target_column is not None
            else []
        )
    )


CATEGORICAL_FEATURES = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
]

CYCLICAL_FEATURES: Dict[str, int] = {}

NUMERICAL_FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
]


def get_feature_config(engineered_df: pl.LazyFrame) -> Dict[str, FeatureConfig]:
    """
    Compute the features config from the feature engineered dataset.
    """
    config: Dict[str, FeatureConfig] = {}

    for column_name in CATEGORICAL_FEATURES:
        unique_values = cast(
            List[str],
            engineered_df.groupby(column_name)
            .agg([])
            .sort(column_name)
            .collect()[column_name]
            .to_list(),
        )
        config[column_name] = {
            "type": "categorical",
            "values": unique_values,
        }

    for column_name in NUMERICAL_FEATURES:
        mean: float
        avg: float
        mean, avg = (
            engineered_df.select(
                [
                    pl.mean(column_name).alias("mean"),
                    pl.std(column_name).alias("std"),
                ]
            )
            .collect()
            .row(0)
        )
        config[column_name] = {
            "type": "numerical",
            "mean": mean,
            "std": avg,
        }

    for column_name, period in CYCLICAL_FEATURES.items():
        config[column_name] = {
            "type": "cyclical",
            "period": period,
        }

    return config


def fit(input: Path, target_column: str, config: Path) -> None:
    """
    Fit the feature engineering pipeline on the training set,
    producing a configuration file that can be used later to transform
    this dataset or inference datasets.
    """
    raw_df = scan_raw_dataset(input)
    engineered_df = input_feature_engineering(raw_df, target_column)
    configuration = get_feature_config(engineered_df)
    write_json(
        config,
        configuration,
    )
