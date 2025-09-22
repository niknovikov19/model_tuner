from typing import Callable, Dict, Tuple

import numpy as np
import xarray as xr

from ..inputs import PopInputND, NetInputND
from ..regimes import PopRegime1D, NetRegime1D

from ..ir_mappers import PopIRMapper, NetIRMapper

from ..map_funcs import MapFuncType, MapFitParams
from ..map_funcs import create_map_func_by_type

from ..slicers import Slicer

from model_tuner.utils import contour_intersections


class PopIRMapper1DInterp(PopIRMapper):

    def __init__(self):
        self.maps2d = None
        self.interp_method = None
        self.interp_params = None
    
    def set_data(self, D: xr.Dataset,
                 interp_method: str = 'vmin_eq',
                 interp_params: dict = None):
        self.maps2d = D.copy()
        if interp_params is None:
            if interp_method == 'vmin_eq':
                interp_params = {'vmin_vals': [-150, -180, -100]}
        self.interp_method = interp_method
        self.interp_params = interp_params
    
    def I_to_R(self, I: PopInputND) -> PopRegime1D:
        R = self.maps2d['rate']
        r = R.interp(**I.vars).item()
        return PopRegime1D(r)
    
    def R_to_I(self, R: PopRegime1D) -> PopInputND:
        if self.interp_method == 'vmin_eq':
            return self._R_to_I_vmin_eq(R)
        raise NotImplementedError(
            f'R_to_I not implemented for interp_method={self.interp_method}')
    
    def _R_to_I_vmin_eq(self, R: PopRegime1D) -> PopInputND:
        for vmin0 in self.interp_params['vmin_vals']:
            pts = contour_intersections(
                self.maps2d['rate'],
                self.maps2d['v_med_min'],
                R.value, vmin0,
                out_column_names=self.maps2d['rate'].dims[::-1]
            )
            if len(pts) != 0:
                return PopInputND(pts.loc[0].to_dict())
        raise ValueError(f'No intersection found for R={R.value}')


class NetIRMapper1DInterp(NetIRMapper):
    
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
