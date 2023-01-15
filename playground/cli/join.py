from pathlib import Path

import click
import polars as pl

from ..logs import setup_logging


@click.command()
@click.option(
    "--input-file",
    "input_files",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    multiple=True,
)
@click.option(
    "--output-file",
    type=click.Path(writable=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
def main(input_files: list[Path], output_file: Path) -> None:
    # Read the input files
    df = pl.concat([pl.read_csv(f) for f in input_files])
    # Write the output file
    df.write_csv(output_file)


if __name__ == "__main__":
    setup_logging()
    main()
