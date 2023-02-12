import logging
import multiprocessing as mp
from pathlib import Path
from typing import ClassVar

import keras_tuner as kt
import numpy as np
import polars as pl
import tensorflow as tf
from matplotlib import pyplot as plt
from returns.curry import partial
from sklearn import metrics

from ..feature_engineering import Features
from ..feature_engineering.config import Summary
from ..models.binary_classification import find_best_threshold
from ..models.class_weights import get_class_weights
from ..models.datasets import get_datasets
from ..models.evaluation import get_ground_truth
from ..models.predict import predict
from . import ModelCustomizationInterface
from .s03e04 import get_count_series

logger = logging.getLogger(__name__)


class S03E04AutoencoderModelCustomization:
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
        id_features: list[pl.Series | pl.Expr] = [pl.col(cls.id_column_name)]
        pca_features: list[pl.Series | pl.Expr] = [
            pl.col(f"V{i}").alias(f"pca_{i}") for i in range(1, 28 + 1)
        ]
        other_features: list[pl.Series | pl.Expr] = [
            pl.col("Amount").alias("amount"),
            pl.col("Time").alias("time"),
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
            },
            categorical_features=set(),
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
    def build_autoencoder_model(
        cls, hp: kt.HyperParameters, no_features: int
    ) -> tf.keras.Model:
        input = tf.keras.layers.Input(
            shape=(no_features,), name="representation", dtype=tf.float32
        )

        latent_space_size = 16
        dropout_rate = 0.5

        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(latent_space_size, activation="linear"),
            ],
            name="encoder",
        )
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(no_features, activation="linear"),
            ],
            name="decoder",
        )
        latent_representation = encoder(input)
        reconstructed = decoder(latent_representation)
        model = tf.keras.Model(
            inputs=input,
            outputs={
                "reconstructed": reconstructed,
                "latent_representation": latent_representation,
            },
        )
        model.compile(
            optimizer="adam",
            loss={
                "reconstructed": tf.keras.losses.MeanAbsoluteError(name="mae"),
                "latent_representation": lambda y_true, y_pred: 0.0,
            },
        )
        return model

    @classmethod
    def get_autoencoder_datasets(
        cls, ds: tf.data.Dataset, *, batch_size: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        unbatched_ds = ds.unbatch()
        legit_ds = unbatched_ds.filter(lambda x, y: tf.equal(y, 0)).batch(batch_size)
        fraud_ds = unbatched_ds.filter(lambda x, y: tf.equal(y, 1)).batch(batch_size)
        stacked_legit_ds = legit_ds.map(
            lambda x, y: tf.stack(
                [value for name, value in x.items() if name != cls.id_column_name],
                axis=1,
            )
        )
        stacked_fraud_ds = fraud_ds.map(
            lambda x, y: tf.stack(
                [value for name, value in x.items() if name != cls.id_column_name],
                axis=1,
            )
        )
        autoenc_legit_ds = stacked_legit_ds.map(lambda x: (x, x))
        autoence_fraud_ds = stacked_fraud_ds.map(lambda x: (x, x))
        return autoenc_legit_ds, autoence_fraud_ds

    @classmethod
    def get_classification_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        stacked_ds = ds.map(
            lambda x, y: (
                tf.stack(
                    [value for name, value in x.items() if name != cls.id_column_name],
                    axis=-1,
                ),
                tf.expand_dims(y, axis=-1),
            )
        )
        return stacked_ds

    @classmethod
    def plot_learning_curves(
        cls, history: tf.keras.callbacks.History, target_path: Path
    ) -> None:
        train_loss_history = history.history["loss"]
        val_loss_history = history.history["val_loss"]
        plt.figure()
        plt.plot(train_loss_history, label="Training Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.legend()
        plt.savefig(target_path)

    @classmethod
    def train_autoencoder(
        cls,
        *,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        autoencoder_model: tf.keras.Model,
        no_features: int,
        batch_size: int,
        model_directory: Path,
    ) -> tf.keras.Model:
        train_legit_ds, train_fraud_ds = cls.get_autoencoder_datasets(
            train_ds, batch_size=batch_size
        )
        validation_legit_ds, validation_fraud_ds = cls.get_autoencoder_datasets(
            validation_ds, batch_size=batch_size
        )

        logger.debug("Training autoencoder model...")
        history = autoencoder_model.fit(
            train_legit_ds,
            validation_data=validation_legit_ds,
            epochs=cls.max_epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    verbose=1,
                    restore_best_weights=True,
                ),
            ],
        )
        logger.debug("Autoencoder model training finished")
        cls.plot_learning_curves(history, model_directory / "autoencoder_learning.png")

        logger.debug("Reconstructing legit validation data...")
        legit_reconstructions = autoencoder_model.predict(validation_legit_ds)[
            "reconstructed"
        ]
        logger.debug("Gathering legit validation data...")
        legit_arr = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
            infer_shape=False,
            element_shape=tf.TensorShape([None, no_features]),
        )
        for x, _ in validation_legit_ds:
            legit_arr = legit_arr.write(legit_arr.size(), x)
        legit_representations = legit_arr.concat()
        logger.debug("Calculating legit validation losses...")
        legit_losses = tf.keras.losses.mae(legit_reconstructions, legit_representations)

        logger.debug("Reconstructing fraud validation data...")
        fraud_reconstructions = autoencoder_model.predict(validation_fraud_ds)[
            "reconstructed"
        ]
        logger.debug("Gathering fraud validation data...")
        fraud_arr = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
            infer_shape=False,
            element_shape=tf.TensorShape([None, no_features]),
        )
        for x, _ in validation_fraud_ds:
            fraud_arr = fraud_arr.write(fraud_arr.size(), x)
        fraud_representations = fraud_arr.concat()
        logger.debug("Calculating fraud validation losses...")
        fraud_losses = tf.keras.losses.mae(fraud_reconstructions, fraud_representations)

        logger.debug("Saving losses histogram...")
        fig = plt.figure()
        legit_ax = fig.add_subplot(111)
        fraud_ax = plt.twinx()
        legit_ax.hist(
            legit_losses.numpy(), bins=20, label="Legit", alpha=0.5, color="b"
        )
        fraud_ax.hist(
            fraud_losses.numpy(), bins=20, label="Fraud", alpha=0.5, color="r"
        )
        legit_ax.legend(loc="upper left")
        fraud_ax.legend(loc="upper right")
        plt.savefig(model_directory / "losses_hist.png")

    @classmethod
    def build_classification_wrapper_model(
        cls,
        frozen_autoencoder_model: tf.keras.Model,
        input_spec: dict[str, tf.TensorSpec],
    ) -> tf.keras.Model:
        features = {
            key: spec for key, spec in input_spec.items() if key != cls.id_column_name
        }
        inputs = {
            key: tf.keras.layers.Input(shape=spec.shape[1:], name=key, dtype=spec.dtype)
            for key, spec in features.items()
        }
        all_inputs = tf.stack([input for input in inputs.values()], axis=-1)
        autoencoder_outputs = frozen_autoencoder_model(all_inputs, training=False)
        reconstructed = autoencoder_outputs["reconstructed"]
        latent_representation = autoencoder_outputs["latent_representation"]
        mae = tf.keras.losses.mae(reconstructed, all_inputs)
        expanded_mae = tf.expand_dims(mae, axis=-1)
        mae_and_latent = tf.concat([expanded_mae, latent_representation], axis=-1)
        dropout_rate = 0.5
        classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="classifier",
        )
        output = classifier(mae_and_latent)
        wrapper_model = tf.keras.Model(inputs=inputs, outputs=output)
        wrapper_model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return wrapper_model

    @classmethod
    def train_classification_model(
        cls,
        *,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        wrapper_model: tf.keras.Model,
        model_directory: Path,
        class_weight: dict[float, float],
    ) -> None:
        history = wrapper_model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=cls.max_epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=3,
                    verbose=1,
                    restore_best_weights=True,
                ),
            ],
            class_weight=class_weight,
        )
        cls.plot_learning_curves(
            history, model_directory / "classification_learning.png"
        )

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
        input_spec, _label_spec = datasets.train.element_spec
        no_features = len(set(input_spec) - {cls.id_column_name})

        best_hps: kt.HyperParameters | None = None
        best_hps = kt.HyperParameters()

        logger.debug("Building autoencoder model...")
        autoencoder_model = cls.build_autoencoder_model(best_hps, no_features)

        model_directory.mkdir(exist_ok=True)

        logger.debug("Training autoencoder model...")
        cls.train_autoencoder(
            train_ds=datasets.train,
            validation_ds=datasets.validation,
            model_directory=model_directory,
            autoencoder_model=autoencoder_model,
            no_features=no_features,
            batch_size=batch_size,
        )

        logger.debug("Creating wrapper model...")
        # freeze the trained model
        autoencoder_model.trainable = False
        wrapper_model = cls.build_classification_wrapper_model(
            frozen_autoencoder_model=autoencoder_model, input_spec=input_spec
        )
        class_weight = get_class_weights(train_file, cls.engineered_label_column_name)
        logger.debug("Training wrapper model...")
        cls.train_classification_model(
            train_ds=datasets.train,
            validation_ds=datasets.validation,
            model_directory=model_directory,
            wrapper_model=wrapper_model,
            class_weight=class_weight,
        )

        logger.debug("Saving wrapper model to %s", model_directory)
        wrapper_model.save(model_directory)

        logger.debug("Evaluating model...")
        loss, auc = wrapper_model.evaluate(datasets.evaluation)
        logger.info("Eval loss: %.3f, auc: %.3f", loss, auc)

        predictions = np.squeeze(wrapper_model.predict(datasets.evaluation), axis=-1)
        ground_truth = get_ground_truth(datasets.evaluation)

        score = metrics.roc_auc_score(ground_truth, predictions)
        logger.info("Eval ROC AUC score: %.3f", score)

        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="autoencoder + DNN"
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


model_customization: ModelCustomizationInterface = S03E04AutoencoderModelCustomization
