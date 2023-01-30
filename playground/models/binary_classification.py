from typing import Callable

import kerastuner as kt
import numpy as np
import tensorflow as tf
from numpy import typing as npt

from .hidden_layers import get_hidden


def build_model(hp: kt.HyperParameters, inputs: dict[str, tf.Tensor]) -> tf.keras.Model:
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


def find_best_threshold(
    y_true: npt.NDArray[np.float32],
    y_pred: npt.NDArray[np.float32],
    metric: Callable[[npt.NDArray[np.float32], npt.NDArray[np.float32]], float],
    n_thresholds: int = 100,
) -> float:
    thresholds = np.linspace(0, 1, n_thresholds)
    scores: list[float] = []
    for threshold in thresholds:
        y_pred_thresholded = (y_pred > threshold).astype(np.float32)
        metric_value = metric(y_true, y_pred_thresholded)
        scores.append(metric_value)
    best_threshold: float = thresholds[np.argmax(scores)]
    return best_threshold
