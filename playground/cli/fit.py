import logging
from pathlib import Path

import click

from ..feature_engineering.config import Configuration
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
    engineered_df = customization.feature_engineering(raw_df)
    logger.debug("Engineering summaries")
    summaries = customization.get_summaries(engineered_df)
    logger.debug("Applying summaries")
    engineered_df = customization.apply_summaries(engineered_df, summaries)
    logger.debug("Getting features config")
    features = customization.features(engineered_df)
    logger.debug("Fitting feature engineering")
    features_configuration = get_feature_config(engineered_df, features)
    logger.debug("Saving config to %s", config_file)
    configuration: Configuration = {
        "features": features_configuration,
        "summaries": summaries,
    }
    write_json(
        config_file,
        configuration,
    )
    logger.info("Done")


if __name__ == "__main__":
    setup_logging()
    main()
