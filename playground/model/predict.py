import logging
from pathlib import Path

import polars as pl
import tensorflow as tf

logger = logging.getLogger(__name__)


def predict(*, model_directory: Path, input: Path, output: Path) -> None:
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
    id_input = tf.keras.Input(shape=(), name="id", dtype=tf.int64)
    inputs = {"id": id_input, **model.input}
    outputs = {"id": id_input, "stroke": tf.squeeze(model.output, axis=-1)}
    wrapper_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    wrapper_model.compile()

    logger.debug("Inference...")
    predictions = wrapper_model.predict(ds)
    logger.debug("Saving predictions to %s", output)
    df = pl.DataFrame(predictions)
    df.write_csv(output)
    logger.debug("Done")
