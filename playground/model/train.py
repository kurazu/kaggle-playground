import logging
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .calibration import get_flat_ground_truth
from .class_weight import get_class_weights

logger = logging.getLogger(__name__)


def get_hidden(
    all_inputs: tf.Tensor,
    *,
    layers: int,
    first_layer_units: int,
    dropout: float,
) -> tf.Tensor:
    hidden = all_inputs
    units = first_layer_units
    for i in range(layers):
        hidden = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(
            hidden
        )
        hidden = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(hidden)
        units //= 2
    return hidden


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

    inputs_spec: Dict[str, tf.TensorSpec]
    inputs_spec, labels_spec = train_ds.element_spec

    input_feature_names = set(inputs_spec) - {"id"}
    inputs = {
        name: tf.keras.layers.Input(shape=(), name=name, dtype=tf.float32)
        for name in input_feature_names
    }
    all_inputs = tf.stack(list(inputs.values()), axis=-1)

    dropout = 0.5
    layers = 3
    first_layer_units = 1024
    hidden = get_hidden(
        all_inputs, layers=layers, first_layer_units=first_layer_units, dropout=dropout
    )

    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )

    class_weight = get_class_weights(train_file)
    logger.debug("Training model with class weights: %s", class_weight)
    # make a positive sample more important than a negative sample
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=20,
        verbose=1,
        callbacks=[
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
        ],
        class_weight=class_weight,
    )
    logger.info("Model trained")

    logger.debug("Saving model to %s", model_directory)
    model.save(model_directory)

    logger.debug("Evaluating model...")
    loss, accuracy, auc = model.evaluate(eval_ds)
    logger.info("Eval loss: %.3f, accuracy: %.3f, auc: %.3f", loss, accuracy, auc)

    predictions = np.squeeze(model.predict(eval_ds), axis=-1)
    ground_truth = tf.cast(get_flat_ground_truth(eval_ds), tf.float32).numpy()

    score = roc_auc_score(ground_truth, predictions)
    logger.info("Eval ROC AUC score: %.3f", score)