from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import numpy as np

from defs_base import PopInput, NetInput
from defs_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0

@dataclass
class PopInputWC(PopInput):
    I: float = 0


@dataclass
class NetRegimeWC(NetRegime):
    def __init__(self, pop_names: List[str] = None,
                 pop_rates: List[float] = None):
        self.pop_regimes = {}
        if pop_names is not None:
            for pop_name, r in zip(pop_names, pop_rates):
                self.pop_regimes[pop_name] = PopRegimeWC(r=r)
    
    def get_pop_rate(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].r
    
    def get_pop_rates_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('r')

@dataclass
class NetInputWC(NetInput):
    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('I')


@dataclass
class NetRegimeListWC(NetRegimeList):
    def get_pop_rates_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('r')
