from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegime1D(PopRegime):
    value: float = 0
    
    def is_valid(self) -> bool:
        return not np.isnan(self.value)
    
    def mix_with(self, R: 'PopRegime1D', alpha) -> None:
        self.value = (1 - alpha) * self.value + alpha * R.value


@dataclass
class NetRegime1D(NetRegime):    

    def get_pop_regime_val(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].value
    
    def get_pop_regimes_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('value')
    
    @classmethod
    def from_values(
            cls,
            pop_names: List[str],
            pop_values: List[float]
            ) -> 'NetRegime1D':
        R = NetRegime1D()
        for pop_name, val in zip(pop_names, pop_values):
            R.pop_regimes[pop_name] = PopRegime1D(value=val)
        return R
    
    @classmethod
    def from_dict(
            cls,
            pop_vals_dict: Dict[str, float]
            ) -> 'NetRegime1D':
        return cls.from_values(
            pop_names=list(pop_vals_dict.keys()),
            pop_values=list(pop_vals_dict.values())
        )
    
    @classmethod
    def mix(cls, R1: 'NetRegime1D', R2: 'NetRegime1D', alpha: float) -> 'NetRegime1D':
        R = deepcopy(R1)
        for pop_name in R.pop_regimes:
            R.pop_regimes[pop_name].mix_with(R2.pop_regimes[pop_name], alpha)
        return R


@dataclass
class NetRegime1DList(NetRegimeList):
    
    def get_pop_regimes_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('value')
    
    @classmethod
    def from_regimes_mat(
            cls,
            pop_names: List[str],
            regimes_mat: np.ndarray  # pops x regimes
            ) -> 'NetRegime1DList':
        L = NetRegime1DList()
        for n in range(regimes_mat.shape[1]):
            regime_vals = regimes_mat[:, n]
            L.net_regimes.append(
                NetRegime1D.from_values(pop_names, regime_vals)
            )
        return L
    
    @classmethod
    def mix(
            cls,
            L1: 'NetRegime1DList',
            L2: 'NetRegime1DList',
            alpha: float
            ) -> 'NetRegime1DList':
        L = NetRegime1DList()
        lst1 = L1.net_regimes
        lst2 = L2.net_regimes
        for R1, R2 in zip(lst1, lst2):
            L.net_regimes.append(NetRegime1D.mix(R1, R2, alpha))
        return L
        
    
