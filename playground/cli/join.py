import logging
from pathlib import Path

import click
import polars as pl

from ..logs import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--old-file",
    "old_file_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--new-file",
    "new_file_path",
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
def main(old_file_path: Path, new_file_path: Path, output_file_path: Path) -> None:
    logger.debug("Reading old file: %s", old_file_path)
    old_df = pl.read_csv(old_file_path)
    logger.debug("Reading new file: %s", new_file_path)
    new_df = pl.read_csv(new_file_path)
    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)
    ignored_columns = old_columns ^ new_columns
    if ignored_columns:
        logger.warning("Will ignore columns: %s", ignored_columns)
    old_df = old_df.select(pl.exclude(list(ignored_columns)))
    new_df = new_df.select(old_df.columns)
    joined_df = pl.concat(
        [
            old_df.with_column(pl.lit("old").alias("source")),
            new_df.with_column(pl.lit("new").alias("source")),
        ]
    )
    logger.debug("Writing joined file: %s", output_file_path)
    joined_df.write_csv(output_file_path)
    logger.info("Wrote joined file: %s", output_file_path)


if __name__ == "__main__":
    setup_logging()
    main()
