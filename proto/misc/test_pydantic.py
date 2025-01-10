from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass
class Type1D:
    value: float

@dataclass
class TypeRate: #(Type1D):
    value: float = Field(alias='rate')
    
    __pydantic_config__ = ConfigDict(allow_population_by_alias=True)


x = TypeRate(rate=10)

print(x)