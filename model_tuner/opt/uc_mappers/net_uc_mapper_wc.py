from typing import Dict, List, Literal

from ..regimes import NetRegimeWC, NetRegimeWCList
from ..map_funcs import MapFunc1DExp, MapFunc1DSigmoid
from ..map_funcs import MapFuncType, create_map_func_by_type, MapFitParams

from .net_uc_mapper import NetUCMapper


class NetUCMapperWC(NetUCMapper):
    def __init__(
            self,
            pop_names: List[str],
            map_type: MapFuncType | str = MapFuncType.EXP_1D,
            map_params: Dict | None = None
            ):
        
        self._pop_names = pop_names
        self._is_identity = True

        map_type = MapFuncType(map_type)
        map_params = map_params or {}
        self._map_funcs = {}
        for pop in self._pop_names:
            self._map_funcs[pop] = create_map_func_by_type(
                map_type, **map_params
            )
    
    def set_to_identity(self):
        self._is_identity = True
    
    def _Ru_to_Rc(self, Ru: NetRegimeWC) -> NetRegimeWC:
        if Ru.get_pop_names() != self._pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if self._is_identity:
            Rc = {pop: Ru.get_pop_rate(pop) for pop in self._pop_names}
        else:
            Rc = {pop: self._map_funcs[pop].apply(Ru.get_pop_rate(pop))
                  for pop in self._pop_names}
        return NetRegimeWC.from_rates_dict(Rc)
        
    def _Rc_to_Ru(self, Rc: NetRegimeWC) -> NetRegimeWC:
        if Rc.get_pop_names() != self._pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        if self._is_identity:
            Ru = {pop: Rc.get_pop_rate(pop) for pop in self._pop_names}
        else:
            Ru = {pop: self._map_funcs[pop].apply_inv(Rc.get_pop_rate(pop))
                  for pop in self._pop_names}
        return NetRegimeWC.from_rates_dict(Ru)
    
    def fit_from_data(self, Ru: NetRegimeWCList, Rc: NetRegimeWCList):
        if len(Ru) != len(Rc):
             raise ValueError('Ru and Rc should have the same length')
        if Ru.get_pop_names() != self._pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if Rc.get_pop_names() != self._pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        
        rr_u_mat = Ru.get_pop_attr_mat('r')
        rr_c_mat = Rc.get_pop_attr_mat('r')

        fit_params = MapFitParams()
        fit_params.par0_kprev = 0 if self._is_identity else 1

        for n, pop in enumerate(self._pop_names):
            self._map_funcs[pop].fit(rr_u_mat[n, :], rr_c_mat[n, :], fit_params)
        self._is_identity = False
