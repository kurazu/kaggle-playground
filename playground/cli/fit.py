import logging
from pathlib import Path

import click

from ..feature_engineering.engineering import get_feature_config
from ..logs import setup_logging
from ..pipelines import load_customization
from ..utils import write_json

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
    "--config-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--customization", "customization_importable_name", type=str, required=True
)
def main(
    train_file: Path, config_file: Path, customization_importable_name: str
) -> None:
    logger.debug("Loading customization %s", customization_importable_name)
    customization = load_customization(customization_importable_name)
    logger.debug("Scanning raw dataset")
    raw_df = customization.scan_raw_dataset(train_file)
    logger.debug("Engineering features")
    engineered_df = customization.feature_engineering(
        raw_df, customization.raw_label_column_name
    )
    logger.debug("Fitting feature engineering")
    configuration = get_feature_config(
        engineered_df,
        categorical_features=customization.categorical_features,
        numerical_features=customization.numerical_features,
        cyclical_features=customization.cyclical_features,
    )
    logger.debug("Saving features config to %s", config_file)
    write_json(
        config_file,
        configuration,
    )
    logger.info("Done")


if __name__ == "__main__":
    setup_logging()
    main()
