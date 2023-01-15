from pathlib import Path

import click

from ..logs import setup_logging
from ..model.predict import predict


@click.command()
@click.option(
    "--input-file",
    type=click.Path(
        exists=True, readable=True, file_okay=True, dir_okay=False, path_type=Path
    ),
    required=True,
)
@click.option(
    "--output-file",
    type=click.Path(writable=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, dir_okay=True, readable=True, path_type=Path),
    required=True,
    default="dnn_saved_model",
)
def main(
    input_file: Path,
    output_file: Path,
    model_dir: Path,
) -> None:
    predict(
        model_directory=model_dir,
        input=input_file,
        output=output_file,
    )


if __name__ == "__main__":
    setup_logging()
    main()
