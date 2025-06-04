from typing import Callable, Dict, Tuple

import numpy as np
import xarray as xr

from ..inputs import PopInputND, NetInputND
from ..regimes import PopRegime1D, NetRegime1D

from ..ir_mappers import PopIRMapper, NetIRMapper

from ..map_funcs import MapFuncType, MapFitParams
from ..map_funcs import create_map_func_by_type

from ..slicers import Slicer


class PopIRMapper1DSlice(PopIRMapper):
    """
    PopIREmpiricalMapper1D is a class that maps input values to regime values and vice versa
    using an empirical mapping function. It also provides functionality to fit the mapping
    function from data.
    Attributes:
        _map_func: The mapping function used to transform input values to regime values and vice versa.
    Methods:
        __init__(map_type: MapFuncType | str = MapFuncType.EXP_1D, map_params: Dict | None = None):
            Initializes the PopIREmpiricalMapper1D with a specified mapping function type and parameters.
        I_to_R(I: PopInput1D) -> PopRegime1D:
            Maps input values (I) to regime values (R) using the mapping function.
        R_to_I(R: PopRegime1D) -> PopInput1D:
            Maps regime values (R) to input values (I) using the inverse of the mapping function.
        fit_from_data(values_in: np.ndarray, values_out: np.ndarray, fit_params: MapFitParams = MapFitParams(),
                      weights: np.ndarray | None = None, bounds: List[Tuple[float, float]] = None) -> None:
            Fits the mapping function to the provided input and output data.
    """
    # TODO: update the comments

    def __init__(
            self,
            slicer: Slicer,
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None,
            ):
        self.slicer = slicer
        map_type = MapFuncType(map_type)
        map_params = map_params or {}
        self._map_func = create_map_func_by_type(map_type, **map_params)
        
    def I_to_R(self, I: PopInputND) -> PopRegime1D:
        raise NotImplementedError(
            'I_to_R method is not applicable for PopIREmpiricalMapper1DSlice'
        )

    def R_to_I(self, R: PopRegime1D) -> PopInputND:
        # Apply 1-d mapping: R to the main coordinate of I
        x_in_main = self._map_func.apply_inv(R.value)
        # Get all coordinates of I (point on the slice)
        x_in = self.slicer.project_1d_coords_to_nd(x_in_main)
        # Convert to PopInputND
        return PopInputND(vars=x_in)
    
    def fit_from_data(
            self,
            X: xr.DataArray,  # n-d
            fit_params: MapFitParams = MapFitParams(),
            weight_func: Callable | None = None,
            bounds: Dict[str, Tuple[float, float]] | None = None
            ) -> None:
        
        # Input: values of the main coordinate of X
        values_in = X.coords[self.slicer.coord_main].values

        # Output: 1-d slice of X
        values_out = self.slicer.get_1d_slice(X, values_in)

        # Weights for fitting
        weights = weight_func(values_out) if weight_func else None

        # Fit 1-d mapping function to the points (values_in, values_out)
        self._map_func.fit(values_in, values_out, fit_params, weights, bounds)
    
    def fit_from_data_1d(
            self,
            values_in: np.ndarray,    # 1-d
            values_out: np.ndarray,   # 1-d
            fit_params: MapFitParams = MapFitParams(),
            weight_func: Callable | None = None,
            bounds: Dict[str, Tuple[float, float]] | None = None
            ) -> None:
        
        if values_in.ndim != 1 or values_out.ndim != 1:
            raise ValueError('Input and output values must be 1-dimensional')
        if len(values_in) != len(values_out):
            raise ValueError('Input and output value vectors should have the same length')
        
        # Weights for fitting
        weights = weight_func(values_out) if weight_func else None
        
        self._map_func.fit(values_in, values_out, fit_params, weights, bounds)


class NetIRMapper1DSlice(NetIRMapper):
    
    def I_to_R(self, I: NetInputND) -> NetRegime1D:
        R = NetRegime1D()
        for name, I_ in I.pop_inputs.items():
            R.pop_regimes[name] = self.pop_IR_mappers[name].I_to_R(I_)
        return R
    
    def R_to_I(self, R: NetRegime1D) -> NetInputND:
        I = NetInputND()
        for name, R_ in R.pop_regimes.items():
            I.pop_inputs[name] = self.pop_IR_mappers[name].R_to_I(R_)
        return I
