import tensorflow as tf


def get_inputs(dataset: tf.data.Dataset, id_column_name: str) -> dict[str, tf.Tensor]:
    inputs_spec: dict[str, tf.TensorSpec]
    inputs_spec, labels_spec = dataset.element_spec

    input_feature_names = set(inputs_spec) - {id_column_name}
    inputs = {
        name: tf.keras.layers.Input(shape=(), name=name, dtype=tf.float32)
        for name in input_feature_names
    }
    return inputs
