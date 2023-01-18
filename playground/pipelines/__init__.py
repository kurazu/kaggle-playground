from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class ModelCustomizationInterface(Protocol):
    def scan_raw_dataset(self, input_file: Path) -> pl.LazyFrame:
        ...

    def feature_engineering(
        self, raw_df: pl.LazyFrame, target_column: str
    ) -> pl.LazyFrame:
        ...

    categorical_features: set[str]
    cyclical_features: dict[str, float]
    numerical_features: set[str]
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
