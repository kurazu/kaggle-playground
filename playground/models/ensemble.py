import tensorflow as tf


def wrap_ensemble(
    inputs: dict[str, tf.Tensor], models: list[tf.keras.Model]
) -> tf.keras.Model:
    concatenated_predictions = tf.keras.layers.Concatenate()(
        [model.output for model in models]
    )
    mean_predictions = tf.reduce_mean(concatenated_predictions, axis=-1, keepdims=True)
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=mean_predictions)
    return ensemble_model
