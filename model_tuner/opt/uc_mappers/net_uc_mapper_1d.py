from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from ..regimes import (
    PopRegime, NetRegime, NetRegimeList,
    PopRegime1D, NetRegime1D, NetRegime1DList
)
from ..map_funcs import MapFuncType, MapFitParams, MapFunc1D
from ..map_funcs import create_map_func_by_type

from .net_uc_mapper import NetUCMapper


class PopUCMapper1D:

    _map_func: MapFunc1D
    _is_identity: bool

    def __init__(
            self,
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None
            ):
        map_type = MapFuncType(map_type)
        map_params = map_params or {}
        self._map_func = create_map_func_by_type(map_type, **map_params)
        self._is_identity = True
    
    def set_to_identity(self):
        self._is_identity = True
    
    def is_valid(self):
        return self._is_identity or self._map_func.is_valid()

    @staticmethod
    def _make_pop_regime_1d(R: PopRegime1D | float) -> PopRegime1D:
        if isinstance(R, PopRegime1D):
            return R   # can be a subclass of PopRegime1D
        else:
            return PopRegime1D(value=float(R))
    
    def _Ru_to_Rc(self, Ru: PopRegime1D | float) -> PopRegime1D:
        Ru = self._make_pop_regime_1d(Ru)
        if self._is_identity:
            x_out = Ru.value
        else:
            x_out = self._map_func.apply(Ru.value)
        return type(Ru)(value=x_out)   # can be a subclass of PopRegime1D

    def _Rc_to_Ru(self, Rc: PopRegime1D | float) -> PopRegime1D:
        Rc = self._make_pop_regime_1d(Rc)
        if self._is_identity:
            x_in = Rc.value
        else:
            x_in = self._map_func.apply_inv(Rc.value)
        return type(Rc)(value=x_in)   # can be a subclass of PopRegime1D
    
    def Ru_to_Rc(
            self,
            Ru: PopRegime1D | float | List[PopRegime1D | float] | np.ndarray
            ) -> PopRegime1D | List[PopRegime1D]:
        if not isinstance(Ru, Iterable):
            Ru = [Ru]
        Rc = []
        for Ru_ in Ru:
            Rc.append(self._Ru_to_Rc(Ru_))
        if len(Rc) == 1:
            Rc = Rc[0]
        return Rc

    def Rc_to_Ru(
            self,
            Rc: PopRegime1D | float | List[PopRegime1D | float] | np.ndarray
            ) -> PopRegime1D | List[PopRegime1D]:
        if not isinstance(Rc, Iterable):
            Rc = [Rc]
        Ru = []
        for Rc_ in Rc:
            Ru.append(self._Rc_to_Ru(Rc_))
        if len(Ru) == 1:
            Ru = Ru[0]
        return Ru        
    
    def fit_from_data(
            self,
            values_in: np.ndarray,    # 1-d
            values_out: np.ndarray,   # 1-d
            fit_params: MapFitParams = MapFitParams(),
            weights: np.ndarray | None = None,  # 1-d
            bounds: Dict[str, Tuple[float, float]] | None = None
            ) -> bool:
                
        if values_in.ndim != 1 or values_out.ndim != 1:
            raise ValueError('Input and output values must be 1-dimensional')
        if len(values_in) != len(values_out):
            raise ValueError('Input and output value vectors should have the same length')
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError('Weights must be 1-dimensional')
            if len(weights) != len(values_in):
                raise ValueError('Value and weight vectors should have the same length')
        
        self._map_func.fit(values_in, values_out, fit_params, weights, bounds)
        self._is_identity = False
        return self._map_func.is_valid()


class NetUCMapper1D(NetUCMapper):

    pop_UC_mappers: Dict[str, PopUCMapper1D]

    def __init__(
            self,
            pop_names: List[str],
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None
            ):
        self.pop_UC_mappers = {}
        for pop in pop_names:
            self.pop_UC_mappers[pop] = PopUCMapper1D(map_type, map_params)
    
    def __getitem__(self, pop_name: str) -> PopUCMapper1D:
        if pop_name not in self.pop_UC_mappers:
            raise KeyError(f"Population '{pop_name}' not found in the mapper")
        return self.pop_UC_mappers[pop_name]

    def __setitem__(self, pop_name: str, mapper: PopUCMapper1D):
        if not isinstance(mapper, PopUCMapper1D):
            raise ValueError("Value must be an instance of PopUCMapper1D")
        self.pop_UC_mappers[pop_name] = mapper
    
    @property
    def pop_names(self):
        return list(self.pop_UC_mappers.keys())
    
    def set_to_identity(self):
        for pop in self.pop_names:
            self.pop_UC_mappers[pop].set_to_identity()
    
    def _Ru_to_Rc(self, Ru: NetRegime1D) -> NetRegime1D:
        if Ru.get_pop_names() != self.pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        Rc = {pop: self.pop_UC_mappers[pop].Ru_to_Rc(Ru[pop])
              for pop in self.pop_names}
        return type(Ru).from_dict(Rc)   # can be a subclass of NetRegime1D
        
    def _Rc_to_Ru(self, Rc: NetRegime1D) -> NetRegime1D:
        if Rc.get_pop_names() != self.pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        Ru = {pop: self.pop_UC_mappers[pop].Rc_to_Ru(Rc[pop])
              for pop in self.pop_names}
        return type(Rc).from_dict(Ru)   # can be a subclass of NetRegime1D
    
    def fit_from_data(
            self,
            Ru: NetRegimeList | List[NetRegime],
            Rc: NetRegimeList | List[NetRegime],
            fit_params: MapFitParams = MapFitParams(),
            weights: np.ndarray | None = None,
            bounds: Dict[str, Tuple[float, float]] | None = None,
            verbose: bool = True
            ) -> bool:
        if len(Ru) != len(Rc):
             raise ValueError('Ru and Rc should have the same length')
        if Ru.get_pop_names() != self.pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if Rc.get_pop_names() != self.pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
            
        rr_u_mat = NetRegime1DList(Ru).get_pop_attr_mat('value')
        rr_c_mat = NetRegime1DList(Rc).get_pop_attr_mat('value')
        
        for n, pop in enumerate(self.pop_names):
            if verbose:
                print(f'Fitting U-C for {pop}...')
            self.pop_UC_mappers[pop].fit_from_data(
                values_in=rr_u_mat[n, :],
                values_out=rr_c_mat[n, :],
                fit_params=fit_params,
                weights=weights,
                bounds=bounds
            )
            
        return all(self.pop_UC_mappers[pop].is_valid()
                   for pop in self.pop_names)
