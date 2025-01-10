from .map_func_1d import MapFunc1D
from .map_func_1d_exp import MapFunc1DExp
from .map_func_1d_sigmoid import MapFunc1DSigmoid


def create_map_func_by_name(func_name: str) -> MapFunc1D:
    if func_name == 'exp_1d':
        return MapFunc1DExp()
    if func_name == 'sigmoid_1d':
        return MapFunc1DSigmoid()
    raise ValueError(f'Unknown map function: {func_name}')