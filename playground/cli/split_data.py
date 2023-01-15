import logging
from pathlib import Path

import click
import polars as pl
from sklearn.model_selection import train_test_split

from ..logs import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
    default="data/train.csv",
)
@click.option(
    "--train-output-file",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
    default="split/train.csv",
)
@click.option(
    "--train-parts",
    type=int,
    required=True,
    default=8,
)
@click.option(
    "--validation-output-file",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
    default="split/validation.csv",
)
@click.option(
    "--validation-parts",
    type=int,
    required=True,
    default=1,
)
@click.option(
    "--evaluation-output-file",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
    default="split/evaluation.csv",
)
@click.option(
    "--evaluation-parts",
    type=int,
    required=True,
    default=1,
)
@click.option(
    "--label-column", type=str, required=True, default="classification_target"
)
def main(
    input_file: Path,
    train_output_file: Path,
    train_parts: int,
    validation_output_file: Path,
    validation_parts: int,
    evaluation_output_file: Path,
    evaluation_parts: int,
    label_column: str,
) -> None:
    df = pl.scan_csv(input_file)
    ids, labels = df.select(["id", label_column]).collect()
    train_ids, rest_ids, train_labels, rest_labels = train_test_split(
        ids,
        labels,
        test_size=(evaluation_parts + validation_parts)
        / (train_parts + evaluation_parts + validation_parts),
        random_state=17,
        stratify=labels,
    )
    validation_ids, evaluation_ids = train_test_split(
        rest_ids,
        test_size=evaluation_parts / (evaluation_parts + validation_parts),
        random_state=17,
        stratify=rest_labels,
    )
    logger.info(
        "Train samples: %d, validation samples: %d, evaluation samples: %d",
        len(train_ids),
        len(validation_ids),
        len(evaluation_ids),
    )
    df.filter(pl.col("id").is_in(train_ids)).collect().write_csv(train_output_file)
    logger.debug("Written train samples to %s", train_output_file)
    df.filter(pl.col("id").is_in(validation_ids)).collect().write_csv(
        validation_output_file
    )
    logger.debug("Written validation samples to %s", validation_output_file)
    df.filter(pl.col("id").is_in(evaluation_ids)).collect().write_csv(
        evaluation_output_file
    )
    logger.debug("Written evaluation samples to %s", evaluation_output_file)


if __name__ == "__main__":
    setup_logging()
    main()
