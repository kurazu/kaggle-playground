from pathlib import Path

import click
import polars as pl
from sklearn.model_selection import train_test_split

from ..feature_engineering.engineering import get_feature_config
from ..feature_engineering.transform import transform_engineered
from ..logs import setup_logging
from ..pipelines.s03e04 import S03E04ModelCustomization as Customization


@click.command()
@click.option(
    "--train-file",
    "train_file_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--old-file",
    "old_file_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--test-file",
    "test_file_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
    required=True,
)
@click.option(
    "--train-output-file",
    "train_output_file_path",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--validation-output-file",
    "validation_output_file_path",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--evaluation-output-file",
    "evaluation_output_file_path",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--test-output-file",
    "test_output_file_path",
    type=click.Path(dir_okay=False, file_okay=True, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--train-parts",
    type=int,
    required=True,
    default=7,
)
@click.option(
    "--validation-parts",
    type=int,
    required=True,
    default=1,
)
@click.option(
    "--evaluation-parts",
    type=int,
    required=True,
    default=1,
)
def main(
    train_file_path: Path,
    old_file_path: Path,
    test_file_path: Path,
    train_output_file_path: Path,
    validation_output_file_path: Path,
    evaluation_output_file_path: Path,
    test_output_file_path: Path,
    train_parts: int,
    validation_parts: int,
    evaluation_parts: int,
) -> None:
    # Load trainig samples from two sources
    raw_train_ds = Customization.scan_raw_dataset(train_file_path).with_columns(
        [pl.lit("train").alias("dataset"), pl.lit("false").alias("predict")]
    )
    raw_old_ds = Customization.scan_raw_dataset(old_file_path).with_columns(
        [
            pl.lit("old").alias("dataset"),
            pl.lit("false").alias("predict"),
            pl.lit(-1, dtype=pl.Int64).alias("id"),
        ]
    )
    assert set(raw_train_ds.columns) == set(raw_old_ds.columns)
    raw_old_ds = raw_old_ds.select(raw_train_ds.columns)
    assert raw_train_ds.columns == raw_old_ds.columns

    # Load test samples and prepend training samples
    raw_test_ds = Customization.scan_raw_dataset(test_file_path).with_columns(
        [pl.lit("train").alias("dataset"), pl.lit("true").alias("predict")]
    )
    assert set(raw_train_ds.columns) > set(raw_test_ds.columns)
    joined_test_ds = prepend_train_to_test(
        raw_test_ds=raw_test_ds, raw_train_ds=raw_train_ds
    )

    # Fit feature engineering on train set
    engineered_train_ds = Customization.feature_engineering(raw_train_ds)
    engineered_old_ds = Customization.feature_engineering(raw_old_ds)
    joined_train_ds = pl.concat([engineered_train_ds, engineered_old_ds])
    train_summaries = Customization.get_summaries(joined_train_ds)
    engineered_joined_train_ds_with_summaries = Customization.apply_summaries(
        joined_train_ds, train_summaries
    )
    features = Customization.features(engineered_joined_train_ds_with_summaries)
    features_configuration = get_feature_config(
        engineered_joined_train_ds_with_summaries, features
    )

    # Transform train set
    transformed_train_ds = transform_engineered(
        engineered_joined_train_ds_with_summaries,
        features_configuration,
        id_column_name=Customization.id_column_name,
        engineered_label_column_name=Customization.engineered_label_column_name,
    ).select(pl.exclude("predict__passthrough"))

    # Split the train set into train, validation, evaluation
    # We want the validation and evaluation sets to be composed only of samples from
    # the original train set (not the old related dataset) to get more meaningful
    # evaluation results.
    transformed_original_ds = transformed_train_ds.filter(
        pl.col("dataset__train__dummy") == 1.0
    )
    ids, labels = transformed_original_ds.select(
        [Customization.id_column_name, Customization.engineered_label_column_name]
    ).collect()
    train_ids, rest_ids, train_labels, rest_labels = train_test_split(
        ids,
        labels,
        test_size=(evaluation_parts + validation_parts)
        / (train_parts + evaluation_parts + validation_parts),
        random_state=17,
        stratify=labels,
    )
    validation_ids, evaluation_ids = train_test_split(
        rest_ids,
        test_size=evaluation_parts / (evaluation_parts + validation_parts),
        random_state=17,
        stratify=rest_labels,
    )
    transformed_train_ds.filter(
        pl.col(Customization.id_column_name).is_in(train_ids)
        | (pl.col("dataset__train__dummy") == 0.0)
    ).collect().write_csv(train_output_file_path)

    transformed_train_ds.filter(
        pl.col(Customization.id_column_name).is_in(validation_ids)
    ).collect().write_csv(validation_output_file_path)

    transformed_train_ds.filter(
        pl.col(Customization.id_column_name).is_in(evaluation_ids)
    ).collect().write_csv(evaluation_output_file_path)

    # Apply feature engineering to the test set
    engineered_joined_test_ds = Customization.feature_engineering(joined_test_ds)
    test_summaries = Customization.get_summaries(engineered_joined_test_ds)
    engineered_joined_test_ds_with_summaries = Customization.apply_summaries(
        engineered_joined_test_ds, test_summaries
    )

    # Transform test set
    transformed_test_ds = transform_engineered(
        engineered_joined_test_ds_with_summaries,
        features_configuration,
        id_column_name=Customization.id_column_name,
        engineered_label_column_name=None,
    )
    transformed_test_ds.filter(pl.col("predict__passthrough") == "true").select(
        pl.exclude("predict__passthrough")
    ).collect().write_csv(test_output_file_path)


def prepend_train_to_test(
    *, raw_test_ds: pl.LazyFrame, raw_train_ds: pl.LazyFrame
) -> pl.LazyFrame:
    raw_train_inputs = raw_train_ds.select(
        raw_test_ds.columns
    )  # omit the target column
    return pl.concat([raw_train_inputs, raw_test_ds])


if __name__ == "__main__":
    setup_logging()
    main()
