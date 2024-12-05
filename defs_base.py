from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import numpy as np


@dataclass
class PopRegime:
    pass

@dataclass      
class PopInput:
    pass


@dataclass
class NetRegime:
    pop_regimes: Dict[str, PopRegime] = field(default_factory=dict)
    
    def __init__(self): pass  # constructor should be defined in children
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(R, attr) for R in self.pop_regimes.values()])

@dataclass
class NetInput:
    pop_inputs: Dict[str, PopInput] = field(default_factory=dict)
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(I, attr) for I in self.pop_inputs.values()])


@dataclass
class NetRegimeList:
    net_regimes: List[NetRegime] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
    
    def __getitem__(self, n: int) -> NetRegime:
        return self.net_regimes[n]
    
    def __setitem__(self, n: int, R: NetRegime):
        self.net_regimes[n] = R
    
    def __len__(self) -> int:
        return len(self.net_regimes)
    
    def copy(self) -> 'NetRegimeList':
        return deepcopy(self)
    
    def check_pop_consistency(self) -> bool:
        for R in self.net_regimes:
            if R.get_pop_names() != self.net_regimes[0].get_pop_names():
                return False
        return True
    
    def get_pop_names(self):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
        return self.net_regimes[0].get_pop_names()
    
    def get_pop_attr_mat(self, attr: str) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
        M = [R.get_pop_attr_vec(attr).reshape(-1, 1) for R in self.net_regimes]
        return np.concatenate(M, axis=1)


class ModelDesc:
    def __init__(self): pass
    def get_pop_names(self) -> List[str]: pass
