import kerastuner as kt
import tensorflow as tf


def get_hidden(inputs: tf.Tensor, hp: kt.HyperParameters) -> tf.Tensor:
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
    hidden = inputs
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
