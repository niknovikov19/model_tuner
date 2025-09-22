from enum import Enum

from .map_func_1d import MapFunc1D
from .map_func_1d_exp import MapFunc1DExp
from .map_func_1d_sigmoid import MapFunc1DSigmoid
from .map_func_1d_sigmoid_line import MapFunc1DSigmoidLine
from .map_func_1d_richards import MapFunc1DRichards
from .map_func_1d_rational_2_1 import MapFunc1DRational21
from .map_func_1d_spline import MapFunc1DSpline


class MapFuncType(Enum):
    EXP_1D = 'exp_1d'
    SIGMOID_1D = 'sigmoid_1d'
    SIGMOID_1D_LINE = 'sigmoid_1d_line'
    RICHARDS_1D = 'richards_1d'
    RATIONAL_1D_2_1 = 'rational_1d_2_1'
    SPLINE_1D = 'spline_1d'

def create_map_func_by_type(func_type: MapFuncType, *args, **kwargs) -> MapFunc1D:
    if func_type == MapFuncType.EXP_1D:
        return MapFunc1DExp(*args, **kwargs)
    if func_type == MapFuncType.SIGMOID_1D:
        return MapFunc1DSigmoid(*args, **kwargs)
    if func_type == MapFuncType.SIGMOID_1D_LINE:
        return MapFunc1DSigmoidLine(*args, **kwargs)
    if func_type == MapFuncType.RICHARDS_1D:
        return MapFunc1DRichards(*args, **kwargs)
    if func_type == MapFuncType.RATIONAL_1D_2_1:
        return MapFunc1DRational21(*args, **kwargs)
    if func_type == MapFuncType.SPLINE_1D:
        return MapFunc1DSpline(*args, **kwargs)
    raise ValueError(f'Unsupported map type: {func_type}')    

def create_map_func_by_name(func_name: str, *args, **kwargs) -> MapFunc1D:
    return create_map_func_by_type(MapFuncType(func_name), *args, **kwargs)
