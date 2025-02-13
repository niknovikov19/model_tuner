from typing import Dict, List, Literal, Tuple

import numpy as np

from ..inputs import PopInput1D, NetInput1D
from ..regimes import PopRegime1D, NetRegime1D

from ..ir_mappers import PopIRMapper, NetIRMapper

from ..map_funcs import MapFuncType, MapFitParams
from ..map_funcs import create_map_func_by_type


class PopIREmpiricalMapper1D(PopIRMapper):
    def __init__(
            self,
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None,
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
            values_out: np.ndarray,
            fit_params: MapFitParams = MapFitParams(),
            weights: np.ndarray | None = None,
            bounds: List[Tuple[float, float]] = None,
            ) -> None:
        if len(values_in) != len(values_out):
             raise ValueError('Input and output value vectors should have the same length')
        if weights and (len(weights) != len(values_in)):
             raise ValueError('Value and weight vectors should have the same length')
        self._map_func.fit(values_in, values_out, fit_params, weights, bounds)


class NetIREmpiricalMapper1D(NetIRMapper):
    
    def I_to_R(self, I: NetInput1D) -> NetRegime1D:
        R = NetRegime1D()
        for name, I_ in I.pop_inputs.items():
            R.pop_regimes[name] = self.pop_IR_mappers[name].I_to_R(I_)
        return R
    
    def R_to_I(self, R: NetRegime1D) -> NetInput1D:
        I = NetInput1D()
        for name, R_ in R.pop_regimes.items():
            I.pop_inputs[name] = self.pop_IR_mappers[name].R_to_I(R_)
        return I
