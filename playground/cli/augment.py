from pathlib import Path

import click
import polars as pl

from ..logs import setup_logging


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
    "--column-name",
    type=str,
    required=True,
)
@click.option(
    "--column-value",
    type=str,
    required=True,
)
def main(
    input_file_path: Path, output_file_path: Path, column_name: str, column_value: str
) -> None:
    pl.read_csv(input_file_path).with_column(
        pl.lit(column_value).alias(column_name)
    ).write_csv(output_file_path)


if __name__ == "__main__":
    setup_logging()
    main()
