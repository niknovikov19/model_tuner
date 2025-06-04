from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from model_tuner.utils import copy_or_ref


@dataclass
class PopRegime:    
    def is_valid(self) -> bool:
        return True


class NetRegime:
    pop_regimes: Dict[str, PopRegime]

    def __init__(
            self,
            R: 'NetRegime' | Dict[str, PopRegime] | None = None,
            force_copy: bool = False
            ):
        if R is None:
            self.pop_regimes = {}
        elif isinstance(R, NetRegime):
            self.pop_regimes = copy_or_ref(R.pop_regimes, force_copy)
        elif isinstance(R, dict):
            self.pop_regimes = copy_or_ref(R, force_copy)
        else:
            raise TypeError('Wrong NetRegime initializer type')
        
        self._check()

    def _check(self) -> None:
        for R_ in self.pop_regimes.values():
            if not isinstance(R_, PopRegime):
                raise TypeError(
                    'NetRegime should contain PopRegime objects.'
                )
    
    def is_valid(self) -> bool:
        return all(R.is_valid() for R in self.pop_regimes.values())
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(R, attr) for R in self.pop_regimes.values()])
    
    def copy(self) -> 'NetRegime':
        return deepcopy(self)
    
    def __getitem__(self, pop_name):
        return self.pop_regimes[pop_name]


class NetRegimeList:
    net_regimes: List[NetRegime]

    def __init__(
            self,
            L: 'NetRegimeList' | List[NetRegime] | None = None,
            force_copy: bool = False
            ):
        if L is None:
            self.net_regimes = []
        elif isinstance(L, NetRegimeList):
            self.net_regimes = copy_or_ref(L.net_regimes, force_copy)
        elif isinstance(L, list):
            self.net_regimes = copy_or_ref(L, force_copy)
        else:
            raise TypeError('Wrong NetRegimeList initializer type')
        
        self._check()
    
    def _check(self) -> None:
        for R in self.net_regimes:
            if not isinstance(R, NetRegime):
                raise TypeError(
                    'NetRegime1DList should contain NetRegime1D objects.'
                )
            R._check()
        self._check_pop_consistency()
    
    def __post_init__(self):
        self._check_pop_consistency()
    
    def _check_pop_consistency(self) -> None:
        for R in self.net_regimes:
            if R.get_pop_names() != self.net_regimes[0].get_pop_names():
                raise ValueError(
                    'All entries of NetRegimeList should have the same pops.'
                )
    
    def __getitem__(self, n: int) -> NetRegime:
        return self.net_regimes[n]
        
    def __setitem__(self, n: int, R: NetRegime):
        self.net_regimes[n] = R
        self._check_pop_consistency()
    
    def __len__(self) -> int:
        return len(self.net_regimes)
    
    def __iter__(self):
        return iter(self.net_regimes)
    
    def copy(self) -> 'NetRegimeList':
        return deepcopy(self)
    
    def append(self, R: NetRegime) -> None:
        self.net_regimes.append(R)
        self._check_pop_consistency()
    
    def get_pop_names(self):
        return self.net_regimes[0].get_pop_names()
    
    def get_pop_attr_mat(self, attr: str) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        M = [R.get_pop_attr_vec(attr).reshape(-1, 1) for R in self.net_regimes]
        return np.concatenate(M, axis=1)
