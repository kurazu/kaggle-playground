from pathlib import Path
from typing import Dict

import polars as pl


def get_class_weights(
    train_file: Path, target_column_name: str = "classification_target"
) -> Dict[float, float]:
    df = pl.scan_csv(train_file)
    total_samples, positive_samples, negative_samples = (
        df.select(
            [
                pl.count().alias("total_count"),
                (pl.col(target_column_name) == 1.0).sum().alias("positive_count"),
                (pl.col(target_column_name) == 0.0).sum().alias("negative_count"),
            ]
        )
        .collect()
        .row(0)
    )

    class_weight = {0.0: 1.0, 1.0: negative_samples / positive_samples}
    return class_weight
