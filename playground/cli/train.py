import logging
from pathlib import Path

import click

from ..logs import setup_logging
from ..model.train import train

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--train-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="split/train.transformed.csv",
)
@click.option(
    "--old-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="data/old.transformed.csv",
)
@click.option(
    "--validation-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="split/valid.transformed.csv",
)
@click.option(
    "--evaluation-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="split/eval.transformed.csv",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    required=True,
)
def main(
    train_file: Path,
    old_file: Path,
    validation_file: Path,
    evaluation_file: Path,
    output_dir: Path,
) -> None:
    logger.debug("Starting training")
    train(
        train_file=train_file,
        old_file=old_file,
        validation_file=validation_file,
        evaluation_file=evaluation_file,
        model_directory=output_dir,
    )
    logger.debug("Training finished")


if __name__ == "__main__":
    setup_logging()
    main()
