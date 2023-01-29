import numpy as np
import numpy.typing as npt
import tensorflow as tf


def _get_flat_ground_truth(ds: tf.data.Dataset) -> tf.Tensor:
    """Flatten the ground truth labels from a dataset."""
    return tf.cast(tf.concat([y for _, y in ds], axis=0), tf.bool)


def get_ground_truth(dataset: tf.data.Dataset) -> npt.NDArray[np.float32]:
    """Get the ground truth labels from a dataset."""
    ground_truth: npt.NDArray[np.float32] = tf.cast(
        _get_flat_ground_truth(dataset), tf.float32
    ).numpy()
    return ground_truth
