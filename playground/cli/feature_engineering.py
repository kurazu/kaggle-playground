import logging
from pathlib import Path

import click

from ..feature_engineering.engineering import fit
from ..logs import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--train-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="train.csv",
)
@click.option(
    "--config-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    required=True,
    default="features.json",
)
@click.option("--target-column", type=str, required=True, default="stroke")
def main(train_file: Path, config_file: Path, target_column: str) -> None:
    logger.info(
        "Fitting feature engineering pipeline on file %s "
        "with target column %r and writing config to %s",
        train_file,
        target_column,
        config_file,
    )
    fit(train_file, target_column, config_file)
    logger.info("Done")


if __name__ == "__main__":
    setup_logging()
    main()
