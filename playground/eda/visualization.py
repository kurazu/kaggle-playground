import statistics

import more_itertools as mit
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import typing as npt


def plot_counts_and_target_factor(
    df: pl.DataFrame,
    *,
    target_column_name: str,
    column_name: str,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
) -> None:
    target_values = df.select(pl.col(target_column_name).unique().alias("target")).sort(
        "target"
    )["target"]
    aggs = [pl.count().alias("count")]
    for target_value in target_values:
        aggs.append(
            (
                pl.col(target_column_name)
                .filter(pl.col(target_column_name) == target_value)
                .count()
                / pl.count()
            ).alias(f"count_{target_value}_factor")
        )
    grouped_df = df.groupby([column_name]).agg(aggs).sort(["count"])

    indices = np.arange(len(grouped_df))
    bars = left_ax.barh(indices, grouped_df["count"])
    left_ax.set_yticks(indices, grouped_df[column_name])
    left_ax.bar_label(bars, fmt="%d")
    left_ax.set_title(f"Counts for {column_name}")

    prev_series: pl.Series | None = None
    for target_value in target_values:
        series = grouped_df[f"count_{target_value}_factor"]
        right_ax.barh(
            indices,
            series,
            label=f"{target_column_name}={target_value}",
            left=prev_series,
        )
        prev_series = series
    right_ax.set_yticks(indices, grouped_df[column_name])
    right_ax.legend()
    right_ax.set_title(f"Target factor for {column_name}")


def plot_histogram_and_target_factor(
    df: pl.DataFrame,
    *,
    target_column_name: str,
    column_name: str,
    bins: int | None = None,
    left_ax: plt.Axes,
    right_ax: plt.Axes,
) -> None:
    if bins is None:
        # automatically adjust the number of bins
        n_unique_values = df.select(pl.col(column_name).n_unique()).item()
        bins = min(n_unique_values, 20)

    total_series = df[column_name]
    (hist, brackets, _) = left_ax.hist(total_series, bins=bins)
    left_ax.set_title(f"{column_name} distribution")
    x = list(map(statistics.mean, mit.sliding_window(brackets, 2)))
    bracket_width = brackets[1] - brackets[0]

    target_values = df.select(pl.col(target_column_name).unique().alias("target")).sort(
        "target"
    )["target"]
    series = (
        df.filter(pl.col(target_column_name) == target_value)[column_name]
        for target_value in target_values
    )
    histograms = (np.histogram(series, bins=brackets)[0] for series in series)
    factors = (histogram / hist for histogram in histograms)
    prev_factor: npt.NDArray[np.float64] | None = None
    for target_value, factor in zip(target_values, factors):
        right_ax.bar(
            x,
            factor,
            width=bracket_width,
            label=f"{target_column_name}={target_value}",
            bottom=prev_factor,
        )
        prev_factor = factor
    right_ax.legend()
    right_ax.set_title(f"Target factor for {column_name}")


def plot_columns_and_target_factors(
    df: pl.DataFrame, *, target_column_name: str, excluded_columns: set[str]
) -> None:
    column_types = dict(zip(df.columns, df.dtypes))
    del column_types[target_column_name]
    for column_name in excluded_columns:
        del column_types[column_name]
    column_types

    n_attrs = len(column_types)
    fig = plt.figure(figsize=(20, 4 * n_attrs))
    gs = GridSpec(n_attrs, 2, figure=fig)
    for idx, (column_name, column_type) in enumerate(column_types.items()):
        left_ax = fig.add_subplot(gs[idx, 0])
        right_ax = fig.add_subplot(gs[idx, 1])
        if column_type in {pl.Boolean, pl.Utf8}:
            plot_counts_and_target_factor(
                df,
                target_column_name=target_column_name,
                column_name=column_name,
                left_ax=left_ax,
                right_ax=right_ax,
            )
        else:
            plot_histogram_and_target_factor(
                df,
                target_column_name=target_column_name,
                column_name=column_name,
                left_ax=left_ax,
                right_ax=right_ax,
            )
