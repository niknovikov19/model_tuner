from typing import Dict, Literal

import numpy as np

from ..inputs import PopInput1D
from ..regimes import PopRegime1D

from ..ir_mappers import PopIRMapper, NetIRMapper

from ..map_funcs import MapFuncType
from ..map_funcs import create_map_func_by_type


class PopIREmpiricalMapper1D(PopIRMapper):
    def __init__(
            self,
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None
            ):
        map_type = MapFuncType(map_type)
        map_params = map_params or {}
        self._map_func = create_map_func_by_type(map_type, **map_params)
        
    def I_to_R(self, I: PopInput1D) -> PopRegime1D:
        x_out = self._map_func.apply(I.value)
        return PopRegime1D(value=x_out)

    def R_to_I(self, R: PopRegime1D) -> PopInput1D:
        x_in = self._map_func.apply_inv(R.value)
        return PopInput1D(value=x_in)
    
    def fit_from_data(
            self,
            values_in: np.ndarray,
            values_out: np.ndarray
            ) -> None:
        if len(values_in) != len(values_out):
             raise ValueError('Value vectors should have the same length')
        self._map_func.fit(values_in, values_out)


NetIREmpiricalMapper1D = NetIRMapper