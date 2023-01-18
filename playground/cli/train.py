import logging
import tempfile
from pathlib import Path

import click

from ..logs import setup_logging
from ..pipelines import load_customization

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--train-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--validation-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--evaluation-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--customization", "customization_importable_name", type=str, required=True
)
def main(
    train_file: Path,
    validation_file: Path,
    evaluation_file: Path,
    output_dir: Path,
    customization_importable_name: str,
) -> None:
    customization = load_customization(customization_importable_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug("Will operate with temporary directory %s", temp_dir)
        logger.debug("Starting training")
        customization.train(
            train_file=train_file,
            validation_file=validation_file,
            evaluation_file=evaluation_file,
            model_directory=output_dir,
            temporary_directory=Path(temp_dir),
        )
        logger.debug("Training finished")


if __name__ == "__main__":
    setup_logging()
    main()
