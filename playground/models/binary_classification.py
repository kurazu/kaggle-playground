import kerastuner as kt
import tensorflow as tf

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
