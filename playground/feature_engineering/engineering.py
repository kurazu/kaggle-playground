from typing import cast

import polars as pl

from . import Features
from .config import FeatureConfig


def get_feature_config(
    engineered_df: pl.LazyFrame, features: Features
) -> dict[str, FeatureConfig]:
    """
    Compute the features config from the feature engineered dataset.
    """
    config: dict[str, FeatureConfig] = {}

    for column_name in features.categorical_features:
        unique_values = cast(
            list[str],
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

    for column_name in features.numerical_features:
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

    for column_name, period in features.cyclical_features.items():
        config[column_name] = {
            "type": "cyclical",
            "period": period,
        }

    return config


def get_auto_features(
    df: pl.LazyFrame,
    *,
    id_column_name: str,
    target_column_name: str,
    cyclical_features: dict[str, float] = {},
) -> Features:
    column_types = dict(zip(df.columns, df.dtypes))
    del column_types[id_column_name]
    del column_types[target_column_name]
    for cyclical_feature in cyclical_features:
        del column_types[cyclical_feature]
    categorical_features: set[str] = set()
    numerical_features: set[str] = set()
    for column_name, column_type in column_types.items():
        if column_type in {pl.Utf8, pl.Categorical}:
            categorical_features.add(column_name)
        else:
            numerical_features.add(column_name)
    return Features(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        cyclical_features=cyclical_features,
    )
