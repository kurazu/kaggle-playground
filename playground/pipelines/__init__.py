from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

from ..feature_engineering import Features
from ..feature_engineering.config import Summary


@runtime_checkable
class ModelCustomizationInterface(Protocol):
    def scan_raw_dataset(self, input_file: Path) -> pl.LazyFrame:
        ...

    def feature_engineering(self, raw_df: pl.LazyFrame) -> pl.LazyFrame:
        ...

    def get_summaries(self, engineered_df: pl.LazyFrame) -> dict[str, Summary]:
        ...

    def apply_summaries(
        self, engineered_df: pl.LazyFrame, summaries: dict[str, Summary]
    ) -> pl.LazyFrame:
        ...

    def features(self, engineered_df: pl.LazyFrame) -> Features:
        ...

    raw_label_column_name: str
    engineered_label_column_name: str
    id_column_name: str

    def train(
        self,
        *,
        train_file: Path,
        validation_file: Path,
        evaluation_file: Path,
        model_directory: Path,
        temporary_directory: Path,
    ) -> None:
        ...

    def predict(self, *, model_directory: Path, input: Path, output: Path) -> None:
        ...


def load_customization(importable_name: str) -> ModelCustomizationInterface:
    module_name, class_name = importable_name.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    customization = getattr(module, class_name)
    assert isinstance(customization, ModelCustomizationInterface)
    return customization
