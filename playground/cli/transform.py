import logging
from pathlib import Path

import click

from ..feature_engineering.config import Configuration
from ..feature_engineering.transform import transform_engineered
from ..logs import setup_logging
from ..pipelines import load_customization
from ..utils import read_json

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--config-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--customization", "customization_importable_name", type=str, required=True
)
@click.option(
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    required=True,
)
def main(
    input_file: Path,
    config_file: Path,
    customization_importable_name: str,
    output_file: Path,
) -> None:
    customization = load_customization(customization_importable_name)
    logger.info(
        "Transforming file %s with config %s and writing to %s",
        input_file,
        config_file,
        output_file,
    )
    raw_df = customization.scan_raw_dataset(input_file)
    engineered_df = customization.feature_engineering(raw_df)
    configuration: Configuration = read_json(config_file)
    engineered_df = customization.apply_summaries(
        engineered_df, summaries=configuration["summaries"]
    )
    transformed_df = transform_engineered(
        engineered_df,
        configuration["features"],
        id_column_name=customization.id_column_name,
        label_column_name=customization.engineered_label_column_name
        if (customization.engineered_label_column_name in engineered_df.columns)
        else None,
    )
    materialized_df = transformed_df.collect()
    materialized_df.write_csv(output_file)
    logger.info("Done")


if __name__ == "__main__":
    setup_logging()
    main()
