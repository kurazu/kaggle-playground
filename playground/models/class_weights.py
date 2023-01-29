import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def get_class_weights(
    train_file: Path, engineered_label_column_name: str
) -> dict[float, float]:
    df = pl.scan_csv(train_file)
    total_samples, positive_samples, negative_samples = (
        df.select(
            [
                pl.count().alias("total_count"),
                (pl.col(engineered_label_column_name) == 1.0)
                .sum()
                .alias("positive_count"),
                (pl.col(engineered_label_column_name) == 0.0)
                .sum()
                .alias("negative_count"),
            ]
        )
        .collect()
        .row(0)
    )

    class_weight = {0.0: 1.0, 1.0: negative_samples / positive_samples}
    logger.debug("Class weights: %s", class_weight)
    return class_weight
