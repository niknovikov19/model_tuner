from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import numpy as np

from defs_base import PopInput, NetInput
from defs_base import PopRegime, NetRegime, NetRegimeList
from utils import from_dict_or_dataclass


@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0

@dataclass
class PopInputWC(PopInput):
    I: float = 0


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
class NetInputWC(NetInput):
    def __init__(self, pop_inputs: Dict[str, PopInputWC | dict] = None):
        pop_inputs = pop_inputs or {}
        self.pop_inputs = {
            pop_name: from_dict_or_dataclass(pop_input, PopInputWC)
            for pop_name, pop_input in pop_inputs.items()
        }
        
    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('I')


@dataclass
class NetRegimeListWC(NetRegimeList):
    def get_pop_rates_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('r')


# Test of NetRegimeWC and NetInputWC constructors
if __name__ == '__main__':
    
    pop_regimes1 = {
        'pop1': PopRegimeWC(r=1),
        'pop2': PopRegimeWC(r=2)
    }
    R1 = NetRegimeWC(pop_regimes=pop_regimes1)
    print(R1)
    
    pop_regimes2 = {
        'pop1': {'r': 1},
        'pop2': {'r': 2}
    }
    R2 = NetRegimeWC(pop_regimes=pop_regimes2)
    print(R2)
    
    pop_inputs1 = {
        'pop1': PopInputWC(I=1),
        'pop2': PopInputWC(I=2)
    }
    I1 = NetInputWC(pop_inputs=pop_inputs1)
    print(I1)
    
    pop_inputs2 = {
        'pop1': {'I': 1},
        'pop2': {'I': 2}
    }
    I2 = NetInputWC(pop_inputs=pop_inputs2)
    print(I2)

