from typing import Any, List, Literal, TypedDict, Union

Summary = dict[str, list[Any]]


class PassThroughFeatureConfig(TypedDict):
    type: Literal["passthrough"]


class CategoricalFeatureConfig(TypedDict):
    type: Literal["categorical"]
    values: List[str]


class NumericalFeatureConfig(TypedDict):
    type: Literal["numerical"]
    mean: float
    std: float


class CyclicalFeatureConfig(TypedDict):
    type: Literal["cyclical"]
    period: float


FeatureConfig = Union[
    PassThroughFeatureConfig,
    CategoricalFeatureConfig,
    NumericalFeatureConfig,
    CyclicalFeatureConfig,
]


class Configuration(TypedDict):
    features: dict[str, FeatureConfig]
    summaries: dict[str, Summary]
