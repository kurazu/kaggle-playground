from typing import NamedTuple


class Features(NamedTuple):
    passthrough_features: set[str]
    categorical_features: set[str]
    numerical_features: set[str]
    cyclical_features: dict[str, float]
