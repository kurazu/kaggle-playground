from pathlib import Path

import click

from ..logs import setup_logging
from ..pipelines import load_customization


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
)
@click.option(
    "--customization", "customization_importable_name", type=str, required=True
)
def main(
    input_file: Path,
    output_file: Path,
    model_dir: Path,
    customization_importable_name: str,
) -> None:
    customization = load_customization(customization_importable_name)
    customization.predict(
        model_directory=model_dir,
        input=input_file,
        output=output_file,
    )


if __name__ == "__main__":
    setup_logging()
    main()
