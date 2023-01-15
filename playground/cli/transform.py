import logging
from pathlib import Path
from typing import Optional

import click

from ..feature_engineering.transform import transform
from ..logs import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--config-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="features.json",
)
@click.option("--target-column", type=str, required=False)
@click.option(
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    required=True,
)
def main(
    input_file: Path, config_file: Path, target_column: Optional[str], output_file: Path
) -> None:
    logger.info(
        "Transforming file %s with config %s, target column %r and writing to %s",
        input_file,
        config_file,
        target_column,
        output_file,
    )
    transform(input_file, config_file, target_column, output_file)
    logger.info("Done")


if __name__ == "__main__":
    setup_logging()
    main()
