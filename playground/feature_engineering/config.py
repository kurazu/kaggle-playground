from typing import List, TypedDict, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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
