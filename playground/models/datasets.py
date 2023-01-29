from pathlib import Path
from typing import NamedTuple

import tensorflow as tf


class Datasets(NamedTuple):
    train: tf.data.Dataset
    validation: tf.data.Dataset
    train_and_validation: tf.data.Dataset
    evaluation: tf.data.Dataset


def get_datasets(
    *,
    train_file: Path,
    validation_file: Path,
    evaluation_file: Path,
    batch_size: int,
    label_column_name: str,
    shuffle_buffer_size: int = 1000,
) -> Datasets:
    train_ds = tf.data.experimental.make_csv_dataset(
        str(train_file),
        batch_size,
        label_name=label_column_name,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=17,
    )

    valid_ds = tf.data.experimental.make_csv_dataset(
        str(validation_file),
        batch_size,
        label_name=label_column_name,
        num_epochs=1,
        shuffle=False,
    )
    train_and_valid_ds = tf.data.experimental.make_csv_dataset(
        [str(train_file), str(validation_file)],
        batch_size,
        label_name=label_column_name,
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=17,
    )
    eval_ds = tf.data.experimental.make_csv_dataset(
        str(evaluation_file),
        batch_size,
        label_name=label_column_name,
        num_epochs=1,
        shuffle=False,
    )

    return Datasets(
        train=train_ds,
        validation=valid_ds,
        train_and_validation=train_and_valid_ds,
        evaluation=eval_ds,
    )
