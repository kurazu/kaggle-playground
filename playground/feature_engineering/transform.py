import itertools as it
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, cast

import polars as pl
from unidecode import unidecode

from ..utils import read_json
from .config import (
    CategoricalFeatureConfig,
    CyclicalFeatureConfig,
    FeatureConfig,
    NumericalFeatureConfig,
)
from .engineering import input_feature_engineering
from .raw import scan_raw_dataset


def get_categorical_feature_expr(
    column_name: str, column_config: CategoricalFeatureConfig
) -> Iterable[pl.Expr]:
    for value in column_config["values"]:
        clean_name = (
            unidecode(value).replace(" ", "_").replace("/", "_").replace(",", "_")
        ).lower()
        yield (pl.col(column_name) == value).cast(pl.Float32).alias(
            f"{column_name}__{clean_name}__dummy"
        )


def get_numerical_feature_expr(
    column_name: str, column_config: NumericalFeatureConfig
) -> Iterable[pl.Expr]:
    z_scaled_expr = (
        pl.col(column_name).cast(pl.Float32) - column_config["mean"]
    ) / column_config["std"]
    yield z_scaled_expr.alias(f"{column_name}__z_scaled")


def get_cyclical_feature_expr(
    column_name: str, column_config: CyclicalFeatureConfig
) -> Iterable[pl.Expr]:
    period = column_config["period"]

    sin_expr = (2 * math.pi * pl.col(column_name) / period).sin()
    yield sin_expr.alias(f"{column_name}__sin")

    cos_expr = (2 * math.pi * pl.col(column_name) / period).cos()
    yield cos_expr.alias(f"{column_name}__cos")


def get_exprs(column_name: str, column_config: FeatureConfig) -> Iterable[pl.Expr]:
    type = column_config["type"]
    if type == "categorical":
        yield from get_categorical_feature_expr(
            column_name, cast(CategoricalFeatureConfig, column_config)
        )
    elif type == "numerical":
        yield from get_numerical_feature_expr(
            column_name, cast(NumericalFeatureConfig, column_config)
        )
    else:
        assert type == "cyclical"
        yield from get_cyclical_feature_expr(
            column_name, cast(CyclicalFeatureConfig, column_config)
        )


def transform_engineered(
    engineered_df: pl.LazyFrame,
    config: Dict[str, FeatureConfig],
    has_target: bool,
) -> pl.LazyFrame:
    """
    Transform the feature engineered dataset into a dataset of
    transformed features.
    """

    expressions = it.chain.from_iterable(it.starmap(get_exprs, config.items()))
    expressions = it.chain([pl.col("id")], expressions)
    if has_target:
        expressions = it.chain(
            expressions, [pl.col("classification_target").cast(pl.Float32)]
        )

    return engineered_df.select(list(expressions))


def transform(
    input: Path, config: Path, target_column: Optional[str], output: Path
) -> None:
    """
    Transform the raw input dataset (a training or inference dataset)
    into a dataset of engineered features.
    """
    raw_df = scan_raw_dataset(input)
    engineered_df = input_feature_engineering(raw_df, target_column)
    configuration: Dict[str, FeatureConfig] = read_json(config)
    transformed_df = transform_engineered(
        engineered_df, configuration, target_column is not None
    )
    materialized_df = transformed_df.collect()
    materialized_df.write_csv(output)
