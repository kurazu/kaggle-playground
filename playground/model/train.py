import logging
import tempfile
from pathlib import Path
from typing import Dict, TypedDict

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from returns.curry import partial
from sklearn.metrics import roc_auc_score

from .calibration import get_flat_ground_truth
from .class_weight import get_class_weights

logger = logging.getLogger(__name__)


def get_hidden(all_inputs: tf.Tensor, hp: kt.HyperParameters) -> tf.Tensor:
    layers: int = hp.Int("layers", 1, 3)
    first_layer_units: int = hp.Int("first_layer_units", 32, 2048, step=32)
    dropout: float = hp.Float("dropout", 0.0, 0.5, step=0.1)
    activation: str = hp.Choice(
        "activation", ["relu", "tanh", "sigmoid", "elu", "selu"]
    )
    hidden = all_inputs
    units = first_layer_units
    for i in range(layers):
        hidden = tf.keras.layers.Dense(units, activation=activation)(hidden)
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        units //= 2
    return hidden


def build_model(hp: kt.HyperParameters, inputs: Dict[str, tf.Tensor]) -> tf.keras.Model:
    all_inputs = tf.stack(list(inputs.values()), axis=-1)

    hidden = get_hidden(
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


def train_model(
    *,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    inputs: Dict[str, tf.Tensor],
    class_weight: Dict[float, float],
) -> tf.keras.Model:
    with tempfile.TemporaryDirectory(prefix="keras_tuner") as temp_dir:

        tuner = kt.Hyperband(
            partial(build_model, inputs=inputs),
            objective=kt.Objective("val_auc", direction="max"),
            max_epochs=10,
            factor=3,
            directory=temp_dir,
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

        tuner.search(
            train_ds,
            validation_data=valid_ds,
            epochs=10,
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
            epochs=10,
            verbose=1,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        val_auc_per_epoch = history.history["val_auc"]
        best_epoch = np.argmax(val_auc_per_epoch) + 1

        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=best_epoch,
            verbose=1,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    logger.info("Model trained")
    return model


class HyperParams(TypedDict):
    dropout: float
    layers: int
    first_layer_units: int
    activation: str


def train(
    *,
    train_file: Path,
    validation_file: Path,
    evaluation_file: Path,
    model_directory: Path,
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
        label_name="classification_target",
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=1000,
        shuffle_seed=17,
    )

    valid_ds = tf.data.experimental.make_csv_dataset(
        str(validation_file),
        batch_size,
        label_name="classification_target",
        num_epochs=1,
        shuffle=False,
    )
    eval_ds = tf.data.experimental.make_csv_dataset(
        str(evaluation_file),
        batch_size,
        label_name="classification_target",
        num_epochs=1,
        shuffle=False,
    )
    class_weight = get_class_weights(train_file)

    inputs_spec: Dict[str, tf.TensorSpec]
    inputs_spec, labels_spec = train_ds.element_spec

    input_feature_names = set(inputs_spec) - {"id"}
    inputs = {
        name: tf.keras.layers.Input(shape=(), name=name, dtype=tf.float32)
        for name in input_feature_names
    }

    model = train_model(
        train_ds=train_ds,
        valid_ds=valid_ds,
        inputs=inputs,
        class_weight=class_weight,
    )

    # models = [
    #     train_model(
    #         train_ds=train_ds,
    #         valid_ds=valid_ds,
    #         inputs=inputs,
    #         class_weight=class_weight,
    #         dropout=0.5,
    #         layers=3,
    #         first_layer_units=1024,
    #         activation="relu",
    #     ),
    #     train_model(
    #         train_ds=train_ds,
    #         valid_ds=valid_ds,
    #         inputs=inputs,
    #         class_weight=class_weight,
    #         dropout=0.5,
    #         layers=3,
    #         first_layer_units=2048,
    #         activation="selu",
    #     ),
    #     train_model(
    #         train_ds=train_ds,
    #         valid_ds=valid_ds,
    #         inputs=inputs,
    #         class_weight=class_weight,
    #         dropout=0.25,
    #         layers=2,
    #         first_layer_units=512,
    #         activation="relu",
    #     ),
    # ]

    # concatenated_predictions = tf.keras.layers.Concatenate()(
    #     [model.output for model in models]
    # )
    # mean_predictions = tf.reduce_mean(concatenated_predictions, axis=-1, keepdims=True)
    # ensemble_model = tf.keras.Model(inputs=inputs, outputs=mean_predictions)
    # ensemble_model.compile(
    #     loss="binary_crossentropy",
    #     metrics=["accuracy", tf.keras.metrics.AUC()],
    # )

    logger.debug("Saving model to %s", model_directory)
    model.save(model_directory)

    logger.debug("Evaluating model...")
    loss, accuracy, auc = model.evaluate(eval_ds)
    logger.info("Eval loss: %.3f, accuracy: %.3f, auc: %.3f", loss, accuracy, auc)

    predictions = np.squeeze(model.predict(eval_ds), axis=-1)
    ground_truth = tf.cast(get_flat_ground_truth(eval_ds), tf.float32).numpy()

    score = roc_auc_score(ground_truth, predictions)
    logger.info("Eval ROC AUC score: %.3f", score)
