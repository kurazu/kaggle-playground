from pathlib import Path

import polars as pl


def scan_raw_dataset(input: Path) -> pl.LazyFrame:
    return pl.scan_csv(input)
