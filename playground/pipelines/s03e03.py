import logging
from pathlib import Path
from typing import ClassVar

import keras_tuner as kt
import numpy as np
import polars as pl
import tensorflow as tf
from returns.curry import partial
from sklearn.metrics import roc_auc_score

from ..feature_engineering import Features
from ..feature_engineering.config import Summary
from ..feature_engineering.engineering import get_auto_features
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
    def get_class_weights(cls, train_file: Path) -> dict[float, float]:
        df = pl.scan_csv(train_file)
        total_samples, positive_samples, negative_samples = (
            df.select(
                [
                    pl.count().alias("total_count"),
                    (pl.col(cls.engineered_label_column_name) == 1.0)
                    .sum()
                    .alias("positive_count"),
                    (pl.col(cls.engineered_label_column_name) == 0.0)
                    .sum()
                    .alias("negative_count"),
                ]
            )
            .collect()
            .row(0)
        )

        class_weight = {0.0: 1.0, 1.0: negative_samples / positive_samples}
        return class_weight

    @classmethod
    def get_hidden(cls, all_inputs: tf.Tensor, hp: kt.HyperParameters) -> tf.Tensor:
        layers: int = hp.Int("layers", 1, 3)
        first_layer_units: int = hp.Choice(
            "first_layer_units", [32, 64, 128, 512, 1024, 2048]
        )
        dropout: float = hp.Float("dropout", 0.0, 0.5, step=0.1)
        activation: str = hp.Choice(
            "activation", ["relu", "tanh", "sigmoid", "elu", "selu"]
        )
        regularization: str | None = hp.Choice(
            "regularization", ["l1", "l2", "l1_l2", "none"]
        )
        if regularization == "none":
            regularization = None
        hidden = all_inputs
        units = first_layer_units
        for i in range(layers):
            hidden = tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularization,
                bias_regularizer=regularization,
            )(hidden)
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
            units //= 2
        return hidden

    @classmethod
    def build_model(
        cls, hp: kt.HyperParameters, inputs: dict[str, tf.Tensor]
    ) -> tf.keras.Model:
        all_inputs = tf.stack(list(inputs.values()), axis=-1)

        hidden = cls.get_hidden(
            all_inputs,
            hp,
        )

        output = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        lr: float = hp.Float("lr", 1e-5, 1e-2, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        return model

    @classmethod
    def train_model(
        cls,
        *,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        train_and_valid_ds: tf.data.Dataset,
        inputs: dict[str, tf.Tensor],
        class_weight: dict[float, float],
        temporary_directory: Path,
    ) -> tf.keras.Model:
        tuner = kt.Hyperband(
            partial(cls.build_model, inputs=inputs),
            objective=kt.Objective("val_auc", direction="max"),
            max_epochs=cls.max_epochs,
            factor=3,
            directory=temporary_directory,
            project_name="playground",
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=1,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                verbose=1,
                restore_best_weights=True,
            ),
        ]
        logger.debug("Starting hyperparameter search")
        tuner.search(
            train_ds,
            validation_data=valid_ds,
            epochs=cls.max_epochs,
            verbose=1,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        (best_hps,) = tuner.get_best_hyperparameters(num_trials=1)
        logger.info("Best hyperparameters: %s", best_hps.values)

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=cls.max_epochs,
            verbose=1,
            class_weight=class_weight,
            callbacks=callbacks,
        )
        val_auc_per_epoch = history.history["val_auc"]
        best_epoch = np.argmax(val_auc_per_epoch) + 1
        logger.info("Best epoch is %d", best_epoch)

        hypermodels: list[tf.keras.Model] = []
        for i in range(3):
            logger.info("Training hypermodel %d", i + 1)
            hypermodel = tuner.hypermodel.build(best_hps)
            hypermodel.fit(
                train_and_valid_ds,
                epochs=best_epoch,
                verbose=1,
                class_weight=class_weight,
                callbacks=[],
            )
            hypermodels.append(hypermodel)

        logger.debug("Wrapping ensemble model")
        ensemble_model = cls.wrap_ensemble(inputs, hypermodels)

        logger.info("Model trained")
        return ensemble_model

    @classmethod
    def wrap_ensemble(
        cls, inputs: dict[str, tf.Tensor], models: list[tf.keras.Model]
    ) -> tf.keras.Model:
        concatenated_predictions = tf.keras.layers.Concatenate()(
            [model.output for model in models]
        )
        mean_predictions = tf.reduce_mean(
            concatenated_predictions, axis=-1, keepdims=True
        )
        ensemble_model = tf.keras.Model(inputs=inputs, outputs=mean_predictions)
        ensemble_model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return ensemble_model

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

        train_ds = tf.data.experimental.make_csv_dataset(
            str(train_file),
            batch_size,
            label_name=cls.engineered_label_column_name,
            num_epochs=1,
            shuffle=True,
            shuffle_buffer_size=1000,
            shuffle_seed=17,
        )

        valid_ds = tf.data.experimental.make_csv_dataset(
            str(validation_file),
            batch_size,
            label_name=cls.engineered_label_column_name,
            num_epochs=1,
            shuffle=False,
        )
        train_and_valid_ds = tf.data.experimental.make_csv_dataset(
            [str(train_file), str(validation_file)],
            batch_size,
            label_name=cls.engineered_label_column_name,
            num_epochs=1,
            shuffle=True,
            shuffle_buffer_size=2000,
            shuffle_seed=17,
        )
        eval_ds = tf.data.experimental.make_csv_dataset(
            str(evaluation_file),
            batch_size,
            label_name=cls.engineered_label_column_name,
            num_epochs=1,
            shuffle=False,
        )
        class_weight = cls.get_class_weights(train_file)

        inputs_spec: dict[str, tf.TensorSpec]
        inputs_spec, labels_spec = train_ds.element_spec

        input_feature_names = set(inputs_spec) - {cls.id_column_name}
        inputs = {
            name: tf.keras.layers.Input(shape=(), name=name, dtype=tf.float32)
            for name in input_feature_names
        }

        model = cls.train_model(
            train_ds=train_ds,
            valid_ds=valid_ds,
            train_and_valid_ds=train_and_valid_ds,
            inputs=inputs,
            class_weight=class_weight,
            temporary_directory=temporary_directory,
        )

        logger.debug("Saving model to %s", model_directory)
        model.save(model_directory)

        logger.debug("Evaluating model...")
        loss, accuracy, auc = model.evaluate(eval_ds)
        logger.info("Eval loss: %.3f, accuracy: %.3f, auc: %.3f", loss, accuracy, auc)

        predictions = np.squeeze(model.predict(eval_ds), axis=-1)
        ground_truth = tf.cast(cls.get_flat_ground_truth(eval_ds), tf.float32).numpy()

        score = roc_auc_score(ground_truth, predictions)
        logger.info("Eval ROC AUC score: %.3f", score)

    @classmethod
    def get_flat_ground_truth(cls, ds: tf.data.Dataset) -> tf.Tensor:
        """Flatten the ground truth labels from a dataset."""
        return tf.cast(tf.concat([y for _, y in ds], axis=0), tf.bool)

    @classmethod
    def predict(cls, *, model_directory: Path, input: Path, output: Path) -> None:
        logger.debug("Loading model from %s", model_directory)
        model = tf.keras.models.load_model(model_directory)
        logger.debug("Inference on %s", input)
        ds = tf.data.experimental.make_csv_dataset(
            str(input),
            64,
            num_epochs=1,
            shuffle=False,
        )
        logger.debug("Building wrapper model...")
        id_input = tf.keras.Input(shape=(), name=cls.id_column_name, dtype=tf.int64)
        inputs = {cls.id_column_name: id_input, **model.input}
        outputs = {
            cls.id_column_name: id_input,
            cls.raw_label_column_name: tf.squeeze(model.output, axis=-1),
        }
        wrapper_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        wrapper_model.compile()

        logger.debug("Inference...")
        predictions = wrapper_model.predict(ds)
        logger.debug("Saving predictions to %s", output)
        df = pl.DataFrame(predictions)
        df.write_csv(output)
        logger.debug("Done")


model_customization: ModelCustomizationInterface = S03E03ModelCustomization
