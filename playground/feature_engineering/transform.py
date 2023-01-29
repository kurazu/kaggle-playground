import itertools as it
import math
import string
from typing import Callable, Dict, Iterable, cast

import polars as pl
from returns.curry import partial
from returns.pipeline import pipe
from unidecode import unidecode

from .config import (
    CategoricalFeatureConfig,
    CyclicalFeatureConfig,
    FeatureConfig,
    NumericalFeatureConfig,
    PassThroughFeatureConfig,
)

ALLOWED_CHARS = set(string.ascii_letters) | set(string.digits)

clean_value: Callable[[str], str] = pipe(  # type: ignore
    str.lower,
    unidecode,
    partial(map, lambda c: c if c in ALLOWED_CHARS else "_"),
    "".join,
)


def get_categorical_feature_expr(
    column_name: str, column_config: CategoricalFeatureConfig
) -> Iterable[pl.Expr]:
    for value in column_config["values"]:
        clean_name = clean_value(value)
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


def get_passthrough_feature_expr(
    column_name: str, column_config: PassThroughFeatureConfig
) -> Iterable[pl.Expr]:
    yield pl.col(column_name).alias(f"{column_name}__passthrough")


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
    elif type == "passthrough":
        yield from get_passthrough_feature_expr(
            column_name, cast(PassThroughFeatureConfig, column_config)
        )
    else:
        assert type == "cyclical"
        yield from get_cyclical_feature_expr(
            column_name, cast(CyclicalFeatureConfig, column_config)
        )


def transform_engineered(
    engineered_df: pl.LazyFrame,
    config: Dict[str, FeatureConfig],
    *,
    id_column_name: str,
    engineered_label_column_name: str | None,
) -> pl.LazyFrame:
    """
    Transform the feature engineered dataset into a dataset of
    transformed features.
    """

    expressions = it.chain.from_iterable(it.starmap(get_exprs, config.items()))
    expressions = it.chain([pl.col(id_column_name)], expressions)
    if engineered_label_column_name is not None:
        expressions = it.chain(
            expressions,
            [
                pl.col(engineered_label_column_name)
                .cast(pl.Float32)
                .alias(engineered_label_column_name)
            ],
        )

    return engineered_df.select(list(expressions))
