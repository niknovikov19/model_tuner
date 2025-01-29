from enum import Enum

from .map_func_1d import MapFunc1D
from .map_func_1d_exp import MapFunc1DExp
from .map_func_1d_sigmoid import MapFunc1DSigmoid


class MapFuncType(Enum):
    EXP_1D = 'exp_1d'
    SIGMOID_1D = 'sigmoid_1d'


def create_map_func_by_type(func_type: MapFuncType, *args, **kwargs) -> MapFunc1D:
    if func_type == MapFuncType.EXP_1D:
        return MapFunc1DExp(*args, **kwargs)
    if func_type == MapFuncType.SIGMOID_1D:
        return MapFunc1DSigmoid(*args, **kwargs)
    raise ValueError(f'Unsupported map type: {func_type}')    

def create_map_func_by_name(func_name: str, *args, **kwargs) -> MapFunc1D:
    return create_map_func_by_type(MapFuncType(func_name), *args, **kwargs)
