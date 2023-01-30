import logging
from pathlib import Path
from typing import Literal, Protocol

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from returns.curry import partial

from .ensemble import wrap_ensemble


class ModelBuiler(Protocol):
    def __call__(
        self, hp: kt.HyperParameters, inputs: dict[str, tf.Tensor]
    ) -> tf.keras.Model:
        ...


logger = logging.getLogger(__name__)


def find_best_hyperparameters(
    *,
    build_model: ModelBuiler,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    inputs: dict[str, tf.Tensor],
    class_weight: dict[float, float],
    temporary_directory: Path,
    max_epochs: int,
    objective_name: str,
    objective_direction: Literal["min", "max"],
) -> kt.HyperParameters:
    logger.debug("Building tuner")
    tuner = kt.Hyperband(
        partial(build_model, inputs=inputs),
        objective=kt.Objective(objective_name, direction=objective_direction),
        max_epochs=max_epochs,
        factor=3,
        directory=temporary_directory,
        project_name="playground",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=objective_name,
            mode=objective_direction,
            patience=3,
            verbose=1,
            restore_best_weights=True,
        ),
    ]
    logger.debug("Starting hyperparameter search")
    tuner.search(
        train_ds,
        validation_data=valid_ds,
        epochs=max_epochs,
        verbose=1,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    (best_hps,) = tuner.get_best_hyperparameters(num_trials=1)
    logger.info("Best hyperparameters: %s", best_hps.values)


def train_model_ensemble(
    *,
    n: int = 3,
    build_model: ModelBuiler,
    best_hps: kt.HyperParameters,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    train_and_valid_ds: tf.data.Dataset,
    inputs: dict[str, tf.Tensor],
    class_weight: dict[float, float],
    max_epochs: int,
    objective_name: str,
    objective_direction: Literal["min", "max"],
) -> tf.keras.Model:
    logger.debug("Training model to determine best epoch")
    model = build_model(best_hps, inputs)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=objective_name,
            mode=objective_direction,
            patience=3,
            verbose=1,
            restore_best_weights=True,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=max_epochs,
        verbose=1,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    objective_per_epoch = history.history[objective_name]
    if objective_direction == "min":
        best_epoch = np.argmin(objective_per_epoch) + 1
    else:
        best_epoch = np.argmax(objective_per_epoch) + 1
    logger.info("Best epoch is %d", best_epoch)

    hypermodels: list[tf.keras.Model] = []
    for i in range(n):
        logger.info("Training hypermodel %d", i + 1)
        hypermodel = build_model(best_hps, inputs)
        hypermodel.fit(
            train_and_valid_ds,
            epochs=best_epoch,
            verbose=1,
            class_weight=class_weight,
            callbacks=[],
        )
        hypermodels.append(hypermodel)

    logger.debug("Wrapping ensemble model")
    ensemble_model = wrap_ensemble(inputs, hypermodels)
    logger.info("Model wrapped")
    return ensemble_model
