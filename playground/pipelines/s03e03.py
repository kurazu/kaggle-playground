import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from ..feature_engineering import Features
from ..feature_engineering.config import Summary
from ..feature_engineering.engineering import get_auto_features
from ..models.binary_classification import build_model
from ..models.class_weights import get_class_weights
from ..models.datasets import get_datasets
from ..models.evaluation import get_ground_truth
from ..models.inputs import get_inputs
from ..models.predict import predict
from ..models.train import train_model
from . import ModelCustomizationInterface

logger = logging.getLogger(__name__)


class S03E03ModelCustomization:
    @classmethod
    def scan_raw_dataset(cls, input_file: Path) -> pl.LazyFrame:
        return pl.scan_csv(input_file)

    @classmethod
    def feature_engineering(cls, raw_df: pl.LazyFrame) -> pl.LazyFrame:
        return raw_df.select(
            [
                pl.col(column_name).alias(cls.engineered_label_column_name)
                if column_name == cls.raw_label_column_name
                else pl.col(column_name)
                for column_name in raw_df.columns
            ]
        ).with_columns(
            [
                (pl.col("MonthlyIncome") / pl.col("Age")).alias("MonthlyIncomePerAge"),
                (pl.col("MonthlyIncome") / (pl.col("TotalWorkingYears") + 1)).alias(
                    "MonthlyIncomePerTotalWorkingYears"
                ),
                (pl.col("MonthlyIncome") / (pl.col("YearsAtCompany") + 1)).alias(
                    "MonthlyIncomePerYearsAtCompany"
                ),
            ]
        )

    @classmethod
    def get_summaries(cls, engineered_df: pl.LazyFrame) -> dict[str, Summary]:
        mean_overtime_per_role = (
            engineered_df.groupby("JobRole")
            .agg(
                [
                    (pl.col("OverTime") == "Yes")
                    .mean()
                    .alias("mean_overtime_per_role"),
                ]
            )
            .collect()
        )
        mean_overtime_per_level = (
            engineered_df.groupby("JobLevel")
            .agg(
                [
                    (pl.col("OverTime") == "Yes")
                    .mean()
                    .alias("mean_overtime_per_level"),
                ]
            )
            .collect()
        )
        mean_salary_per_role_and_level = (
            engineered_df.groupby(["JobRole", "JobLevel"])
            .agg(
                [
                    pl.col("MonthlyIncome")
                    .mean()
                    .alias("mean_salary_per_role_and_level"),
                ]
            )
            .collect()
        )
        return {
            "mean_overtime_per_role": mean_overtime_per_role.to_dict(as_series=False),
            "mean_overtime_per_level": mean_overtime_per_level.to_dict(as_series=False),
            "mean_salary_per_role_and_level": mean_salary_per_role_and_level.to_dict(
                as_series=False
            ),
        }

    @classmethod
    def apply_summaries(
        cls, engineered_df: pl.LazyFrame, summaries: dict[str, Summary]
    ) -> pl.LazyFrame:
        mean_overtime_per_role = pl.DataFrame(
            summaries["mean_overtime_per_role"]
        ).lazy()
        mean_overtime_per_level = pl.DataFrame(
            summaries["mean_overtime_per_level"]
        ).lazy()
        mean_salary_per_role_and_level = pl.DataFrame(
            summaries["mean_salary_per_role_and_level"]
        ).lazy()
        return (
            engineered_df.join(mean_overtime_per_role, on="JobRole", how="left")
            .join(mean_overtime_per_level, on="JobLevel", how="left")
            .join(
                mean_salary_per_role_and_level, on=["JobRole", "JobLevel"], how="left"
            )
        ).with_column(
            (
                (pl.col("MonthlyIncome") - pl.col("mean_salary_per_role_and_level"))
                / pl.col("mean_salary_per_role_and_level")
            ).alias("monthly_income_relative")
        )

    @classmethod
    def features(cls, engineered_df: pl.LazyFrame) -> Features:
        return get_auto_features(
            engineered_df,
            id_column_name=cls.id_column_name,
            target_column_name=cls.engineered_label_column_name,
        )

    id_column_name: ClassVar[str] = "id"
    raw_label_column_name: ClassVar[str] = "Attrition"
    engineered_label_column_name: ClassVar[str] = "classification_target"
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

        model = train_model(
            build_model=build_model,
            train_ds=datasets.train,
            valid_ds=datasets.validation,
            train_and_valid_ds=datasets.train_and_validation,
            inputs=inputs,
            class_weight=class_weight,
            temporary_directory=temporary_directory,
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
        loss, accuracy, auc = model.evaluate(datasets.evaluation)
        logger.info("Eval loss: %.3f, accuracy: %.3f, auc: %.3f", loss, accuracy, auc)

        predictions = np.squeeze(model.predict(datasets.evaluation), axis=-1)
        ground_truth = get_ground_truth(datasets.evaluation)

        score = roc_auc_score(ground_truth, predictions)
        logger.info("Eval ROC AUC score: %.3f", score)

    @classmethod
    def predict(cls, *, model_directory: Path, input: Path, output: Path) -> None:
        predict(
            model_directory=model_directory,
            input=input,
            output=output,
            id_column_name=cls.id_column_name,
            raw_label_column_name=cls.raw_label_column_name,
        )


model_customization: ModelCustomizationInterface = S03E03ModelCustomization
