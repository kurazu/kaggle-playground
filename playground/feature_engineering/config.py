from typing import List, Literal, TypedDict, Union


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
    CategoricalFeatureConfig, NumericalFeatureConfig, CyclicalFeatureConfig
]
