import logging
from pathlib import Path

import click
import polars as pl

from ..logs import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-file",
    "input_file_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--output-file",
    "output_file_path",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--column",
    type=str,
    required=True,
)
def main(input_file_path: Path, output_file_path: Path, column: str) -> None:
    logger.debug("Reading file: %s", input_file_path)
    input_df = pl.scan_csv(input_file_path)
    filtered_df = input_df.select(pl.exclude(column)).collect()
    logger.debug("Writing output file: %s", output_file_path)
    filtered_df.write_csv(output_file_path)
    logger.info("Wrote output file: %s", output_file_path)


if __name__ == "__main__":
    setup_logging()
    main()
