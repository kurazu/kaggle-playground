import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Iterable

import polars as pl

from ..feature_engineering import Features
from ..feature_engineering.config import Summary
from . import ModelCustomizationInterface

logger = logging.getLogger(__name__)


def count_timestamps(series: Iterable[datetime], period: timedelta) -> Iterable[int]:
    past: deque[datetime] = deque()
    for timestamp in series:
        # purge queue of all entries older than current `timestamp` by `period`
        earliest_allowed = timestamp - period
        while past and past[0] < earliest_allowed:
            past.popleft()

        past.append(timestamp)
        yield len(past)


class S03E04ModelCustomization:
    @classmethod
    def scan_raw_dataset(cls, input_file: Path) -> pl.LazyFrame:
        return pl.scan_csv(input_file)

    @classmethod
    def feature_engineering(cls, raw_df: pl.LazyFrame) -> pl.LazyFrame:
        initial_date = pl.datetime(2023, 1, 1)
        materialized_timestamps = raw_df.select(
            (
                (pl.col("Time") * 1000).cast(pl.Duration(time_unit="ms")) + initial_date
            ).alias("ts")
        ).collect()["ts"]
        count_series: list[pl.Series] = []
        for milliseconds in [50, 100, 200, 500, 1000]:
            period = timedelta(milliseconds=milliseconds)
            logger.debug("Counting timestamps: %s", period)
            counts = pl.Series(
                f"count_{milliseconds}",
                count_timestamps(materialized_timestamps, period),
                dtype=pl.Int64,
            )
            logger.debug("Counted timestamps: %s", period)
            count_series.append(counts)
        id_features = [pl.col(cls.id_column_name)]

        pca_features = [pl.col(f"V{i}").alias(f"pca_{i}") for i in range(1, 28 + 1)]
        target_features: list[pl.Series | pl.Expr] = (
            [pl.col(cls.raw_label_column_name)]
            if cls.raw_label_column_name in raw_df
            else []
        )

        return raw_df.select(
            id_features + pca_features + count_series + target_features
        )

    @classmethod
    def get_summaries(cls, engineered_df: pl.LazyFrame) -> dict[str, Summary]:
        return {}

    @classmethod
    def apply_summaries(
        cls, engineered_df: pl.LazyFrame, summaries: dict[str, Summary]
    ) -> pl.LazyFrame:
        assert not summaries
        return engineered_df

    @classmethod
    def features(cls, engineered_df: pl.LazyFrame) -> Features:
        return Features(
            passthrough_features={
                col for col in engineered_df.columns if col.startswith("pca_")
            },
            categorical_features=set(),
            numerical_features={
                col for col in engineered_df.columns if col.startswith("count_")
            },
            cyclical_features={},
        )

    raw_label_column_name: ClassVar[str] = "Class"
    engineered_label_column_name: ClassVar[str] = "target"
    id_column_name: ClassVar[str] = "id"

    @classmethod
    def train(
        cls,
        *,
        train_file: Path,
        validation_file: Path,
        evaluation_file: Path,
        model_directory: Path,
        temporary_directory: Path,
    ) -> None:
        breakpoint()

    @classmethod
    def predict(cls, *, model_directory: Path, input: Path, output: Path) -> None:
        breakpoint()


model_customization: ModelCustomizationInterface = S03E04ModelCustomization
