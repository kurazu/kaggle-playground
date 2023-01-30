import logging
import multiprocessing as mp
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Iterable

import keras_tuner as kt
import numpy as np
import polars as pl
import tensorflow as tf
from matplotlib import pyplot as plt
from returns.curry import partial
from sklearn import metrics

from ..feature_engineering import Features
from ..feature_engineering.config import Summary
from ..models.binary_classification import build_model, find_best_threshold
from ..models.class_weights import get_class_weights
from ..models.datasets import get_datasets
from ..models.evaluation import get_ground_truth
from ..models.inputs import get_inputs
from ..models.predict import predict
from ..models.train import find_best_hyperparameters, train_model_ensemble
from . import ModelCustomizationInterface

logger = logging.getLogger(__name__)


def count_timestamps(series: Iterable[datetime], period: timedelta) -> Iterable[int]:
    """
    Using pl.groupby_rolling uses up too much memory, so we have to do it manually.
    """
    past: deque[datetime] = deque()
    for timestamp in series:
        # purge queue of all entries older than current `timestamp` by `period`
        earliest_allowed = timestamp - period
        while past and past[0] < earliest_allowed:
            past.popleft()

        past.append(timestamp)
        yield len(past)


def get_count_series(
    materialized_timestamps: pl.Series, milliseconds: int
) -> pl.Series:
    period = timedelta(milliseconds=milliseconds)
    logger.debug("Counting timestamps: %s", period)
    counts = pl.Series(
        f"count_{milliseconds}",
        count_timestamps(materialized_timestamps, period),
        dtype=pl.Int64,
    )
    logger.debug("Counted timestamps: %s", period)
    return counts


class S03E04ModelCustomization:
    @classmethod
    def scan_raw_dataset(cls, input_file: Path) -> pl.LazyFrame:
        return pl.scan_csv(input_file, dtypes={"Time": pl.Float32, "id": pl.Int64})

    @classmethod
    def feature_engineering(cls, raw_df: pl.LazyFrame) -> pl.LazyFrame:
        initial_date = pl.datetime(2023, 1, 1)
        materialized_timestamps = raw_df.select(
            (
                (pl.col("Time") * 1000).cast(pl.Duration(time_unit="ms")) + initial_date
            ).alias("ts")
        ).collect()["ts"]
        logger.debug("Starting timestamps aggregation")
        with mp.Pool() as pool:
            count_series = list(
                pool.imap_unordered(
                    partial(get_count_series, materialized_timestamps),
                    [
                        100,  # a tenth of a second
                        500,  # half a second
                        1000,  # a second
                        10000,  # ten seconds
                        30000,  # thirty seconds
                        60000,  # a minute
                        300000,  # five minutes
                        600000,  # ten minutes
                        1800000,  # thirty minutes
                        3600000,  # an hour
                    ],
                )
            )
        logger.debug("Finished timestamps aggregation")
        id_features = [pl.col(cls.id_column_name)]
        pca_features = [pl.col(f"V{i}").alias(f"pca_{i}") for i in range(1, 28 + 1)]
        other_features = [
            pl.col("Amount").alias("amount"),
            pl.col("Time").alias("time"),
            pl.col("dataset"),
            pl.col("predict"),
        ]
        target_features: list[pl.Series | pl.Expr] = (
            [pl.col(cls.raw_label_column_name).alias(cls.engineered_label_column_name)]
            if cls.raw_label_column_name in raw_df
            else []
        )

        return raw_df.select(
            id_features + pca_features + other_features + count_series + target_features
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
            }
            | {"predict"},
            categorical_features={"dataset"},
            numerical_features={
                col for col in engineered_df.columns if col.startswith("count_")
            }
            | {"amount"},
            cyclical_features={
                "time": 60 * 60 * 24,
            },
        )

    raw_label_column_name: ClassVar[str] = "Class"
    engineered_label_column_name: ClassVar[str] = "target"
    id_column_name: ClassVar[str] = "id"
    max_epochs: ClassVar[int] = 20

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
        """Train."""
        logger.debug(
            "Starting model training based on train=%s, validation=%s, evaluation=%s",
            train_file,
            validation_file,
            evaluation_file,
        )
        batch_size = 64

        datasets = get_datasets(
            train_file=train_file,
            validation_file=validation_file,
            evaluation_file=evaluation_file,
            batch_size=batch_size,
            label_column_name=cls.engineered_label_column_name,
        )

        class_weight = get_class_weights(train_file, cls.engineered_label_column_name)

        inputs = get_inputs(datasets.train, cls.id_column_name)

        best_hps: kt.HyperParameters | None = None

        best_hps = kt.HyperParameters()
        best_hps.Fixed("layers", 2)
        best_hps.Fixed("first_layer_units", 64)
        best_hps.Fixed("dropout", 0.1)
        best_hps.Fixed("activation", "relu")
        best_hps.Fixed("regularization", "none")
        best_hps.Fixed("lr", 0.00027255731067274565)

        if best_hps is None:
            best_hps = find_best_hyperparameters(
                build_model=build_model,
                train_ds=datasets.train,
                valid_ds=datasets.validation,
                inputs=inputs,
                class_weight=class_weight,
                temporary_directory=temporary_directory,
                max_epochs=cls.max_epochs,
                objective_name="val_auc",
                objective_direction="max",
            )
        model = train_model_ensemble(
            best_hps=best_hps,
            build_model=build_model,
            train_ds=datasets.train,
            valid_ds=datasets.validation,
            train_and_valid_ds=datasets.train_and_validation,
            inputs=inputs,
            class_weight=class_weight,
            max_epochs=cls.max_epochs,
            objective_name="val_auc",
            objective_direction="max",
        )
        model.compile(
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC()],
        )

        logger.debug("Saving model to %s", model_directory)
        model.save(model_directory)

        logger.debug("Evaluating model...")
        loss, auc = model.evaluate(datasets.evaluation)
        logger.info("Eval loss: %.3f, auc: %.3f", loss, auc)

        predictions = np.squeeze(model.predict(datasets.evaluation), axis=-1)
        ground_truth = get_ground_truth(datasets.evaluation)

        score = metrics.roc_auc_score(ground_truth, predictions)
        logger.info("Eval ROC AUC score: %.3f", score)

        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="DNN ensemble"
        )
        display.plot()
        plt.savefig(model_directory / "roc_curve.png")

        best_threshold = find_best_threshold(
            ground_truth, predictions, metrics.roc_auc_score
        )
        logger.info("Best threshold: %.2f", best_threshold)
        prediction_decisions = (predictions > best_threshold).astype(np.float32)
        count_confusion_matrix = metrics.confusion_matrix(
            ground_truth, prediction_decisions
        )
        metrics.ConfusionMatrixDisplay(
            count_confusion_matrix,
            display_labels=["legit", "fraud"],
        ).plot(cmap="Blues", values_format="d")
        plt.savefig(model_directory / "count_confusion_matrix.png")
        normalized_confusion_matrix = metrics.confusion_matrix(
            ground_truth, prediction_decisions, normalize="true"
        )
        metrics.ConfusionMatrixDisplay(
            normalized_confusion_matrix,
            display_labels=["legit", "fraud"],
        ).plot(cmap="Blues", values_format=".2f")
        plt.savefig(model_directory / "normalized_confusion_matrix.png")

    @classmethod
    def predict(cls, *, model_directory: Path, input: Path, output: Path) -> None:
        predict(
            model_directory=model_directory,
            input=input,
            output=output,
            id_column_name=cls.id_column_name,
            raw_label_column_name=cls.raw_label_column_name,
        )


model_customization: ModelCustomizationInterface = S03E04ModelCustomization
