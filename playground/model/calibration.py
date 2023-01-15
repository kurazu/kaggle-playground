import logging
from pathlib import Path
from typing import List

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def get_flat_ground_truth(ds: tf.data.Dataset) -> tf.Tensor:
    """Flatten the ground truth labels from a dataset."""
    return tf.cast(tf.concat([y for _, y in ds], axis=0), tf.bool)


def calibrate_threshold(
    model: tf.keras.Model, ds: tf.data.Dataset, model_dir: Path
) -> float:
    """
    Find the best classification threshold for given model over the given dataset.

    Stores a chart of the F1 score for different thresholds in the model directory.
    """
    probabilities = tf.squeeze(
        tf.convert_to_tensor(model.predict(ds), dtype=tf.float32), axis=-1
    )
    ground_truth = get_flat_ground_truth(ds)
    thresholds = tf.linspace(0.0, 1.0, 100)
    f1_scores: List[float] = []
    for threshold in thresholds:
        predictions = probabilities > threshold
        f1 = f1_score(ground_truth, predictions, pos_label=True, average="weighted")
        f1_scores.append(f1)
    best_idx = tf.argmax(f1_scores)
    best_threshold: float = thresholds[best_idx].numpy()
    best_score = f1_scores[best_idx]
    logger.info("Best threshold: %.3f, best score: %.3f", best_threshold, best_score)

    plt.plot(thresholds.numpy(), f1_scores, label="F1 score")
    plt.axvline(best_threshold, color="red", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.legend()

    chart_path = model_dir / "calibration.png"
    plt.savefig(chart_path)

    logger.debug("Calibration chart saved to %s", chart_path)

    return best_threshold
