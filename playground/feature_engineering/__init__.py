from typing import NamedTuple


class Features(NamedTuple):
    categorical_features: set[str]
    numerical_features: set[str]
    cyclical_features: dict[str, float]
