from typing import Dict, List, cast

import polars as pl

from .config import FeatureConfig


def get_feature_config(
    engineered_df: pl.LazyFrame,
    *,
    categorical_features: set[str],
    numerical_features: set[str],
    cyclical_features: dict[str, float],
) -> Dict[str, FeatureConfig]:
    """
    Compute the features config from the feature engineered dataset.
    """
    config: Dict[str, FeatureConfig] = {}

    for column_name in categorical_features:
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

    for column_name in numerical_features:
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

    for column_name, period in cyclical_features.items():
        config[column_name] = {
            "type": "cyclical",
            "period": period,
        }

    return config
