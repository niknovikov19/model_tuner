from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegime1D(PopRegime):
    value: float = 0


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


@dataclass
class NetRegime1DList(NetRegimeList):
    
    def get_pop_regimes_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('value')
