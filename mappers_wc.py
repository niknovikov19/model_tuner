from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import numpy as np

from defs_wc import PopInputWC, NetInputWC
from defs_wc import PopRegimeWC, NetRegimeWC, NetRegimeListWC
from map_funcs_1d import MapFunc1DExp, MapFunc1DSigmoid
from mappers_base import PopIRMapper, NetIRMapper, NetUCMapper
from model_wc import PopParamsWC, ModelDescWC, wc_gain, wc_gain_inv


class PopIRMapperWC(PopIRMapper):
    def __init__(self, pop_params: PopParamsWC):
        self.pop_params = pop_params
        
    def I_to_R(self, I: PopInputWC) -> PopRegimeWC:
        r = wc_gain(I.I, self.pop_params)
        return PopRegimeWC(r=r)

    def R_to_I(self, R: PopRegimeWC) -> PopInputWC:
        I = wc_gain_inv(R.r, self.pop_params)
        return PopInputWC(I=I)

class NetIRMapperWC(NetIRMapper):
    def __init__(self, model: ModelDescWC):
        super().__init__()
        for name, pop in model.pops.items():
            self.set_pop_mapper(name, PopIRMapperWC(pop))


class NetUCMapperWC(NetUCMapper):
    def __init__(
            self,
            pop_names: List[str],
            map_type: Literal['exp', 'sigmoid'] = 'exp'):
        self._pop_names = pop_names
        self._is_identity = True
        self._map_funcs = {}
        for pop in self._pop_names:
            if map_type == 'exp':
                self._map_funcs[pop] = MapFunc1DExp()
            elif map_type == 'sigmoid':
                self._map_funcs[pop] = MapFunc1DSigmoid()
            else:
                raise ValueError(f'Unknown map type: {map_type}')
    
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
    
    def fit_from_data(self, Ru: NetRegimeListWC, Rc: NetRegimeListWC):
        if len(Ru) != len(Rc):
             raise ValueError('Ru and Rc should have the same length')
        if Ru.get_pop_names() != self._pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if Rc.get_pop_names() != self._pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        rr_u_mat = Ru.get_pop_attr_mat('r')
        rr_c_mat = Rc.get_pop_attr_mat('r')
        from_prev = not self._is_identity
        for n, pop in enumerate(self._pop_names):
            self._map_funcs[pop].fit(rr_u_mat[n, :], rr_c_mat[n, :], from_prev)
        self._is_identity = False
