from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from model_tuner.utils import from_dict_or_dataclass

from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0


@dataclass
class NetRegimeWC(NetRegime):    
    def __init__(self, pop_regimes: Dict[str, PopRegimeWC | dict] = None):
        pop_regimes = pop_regimes or {}
        self.pop_regimes = {
            pop_name: from_dict_or_dataclass(pop_regime, PopRegimeWC)
            for pop_name, pop_regime in pop_regimes.items()
        }

    def get_pop_rate(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].r
    
    def get_pop_rates_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('r')
    
    @classmethod
    def from_rates(cls, pop_names: List[str], pop_rates: List[float]) -> 'NetRegimeWC':
        R = NetRegimeWC()
        for pop_name, r in zip(pop_names, pop_rates):
            R.pop_regimes[pop_name] = PopRegimeWC(r=r)
        return R
    
    @classmethod
    def from_rates_dict(cls, pop_rates: Dict[str, float]) -> 'NetRegimeWC':
        return cls.from_rates(
            pop_names=list(pop_rates.keys()),
            pop_rates=list(pop_rates.values())
        )


@dataclass
class NetRegimeWCList(NetRegimeList):
    def get_pop_rates_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('r')
