from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from model_tuner.utils import from_dict_or_dataclass

from .regime_base import PopRegime, NetRegime, NetRegimeList
from .regime_1d import PopRegime1D, NetRegime1D, NetRegime1DList


@dataclass        
class PopRegimeWC(PopRegime1D):
    @property
    def r(self) -> float:
        return self.value
    
    @r.setter
    def r(self, new_value: float) -> None:
        self.value = new_value


@dataclass
class NetRegimeWC(NetRegime1D):    
    """ def __init__(self, pop_regimes: Dict[str, PopRegimeWC | dict] = None):
        pop_regimes = pop_regimes or {}
        self.pop_regimes = {
            pop_name: from_dict_or_dataclass(pop_regime, PopRegimeWC)
            for pop_name, pop_regime in pop_regimes.items()
        } """

    def get_pop_rate(self, pop_name: str) -> float:
        return self.get_pop_regime_val(pop_name)
    
    def get_pop_rates_vec(self) -> np.ndarray:
        return self.get_pop_regimes_vec()
    
    @classmethod
    def _convert_parent(cls, R: NetRegime) -> 'NetRegimeWC':
        R.__class__ = NetRegimeWC
        for r in R.pop_regimes.values():
            r.__class__ = PopRegimeWC
        return R

    @classmethod
    def from_rates(
            cls,
            pop_names: List[str],
            pop_rates: List[float]
            ) -> 'NetRegimeWC':
        R = NetRegime1D.from_values(pop_names, pop_rates)
        return cls._convert_parent(R)
    
    @classmethod
    def from_rates_dict(
            cls,
            pop_rates: Dict[str, float]
            ) -> 'NetRegimeWC':
        return cls.from_rates(
            pop_names=list(pop_rates.keys()),
            pop_rates=list(pop_rates.values())
        )


@dataclass
class NetRegimeWCList(NetRegime1DList):

    def get_pop_rates_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_regimes_mat()
    
    @classmethod
    def _convert_parent(cls, L: NetRegimeList) -> 'NetRegimeWCList':
        L.__class__ = NetRegimeWCList
        for R in L.net_regimes:
            NetRegimeWC._convert_parent(R)
        return L

    @classmethod
    def from_rates_mat(
            cls,
            pop_names: List[str],
            rates_mat: np.ndarray  # pops x regimes
            ) -> 'NetRegimeWCList':
        """Creates from (pops x regimes) matrix. """
        L = NetRegime1DList.from_regimes_mat(pop_names, rates_mat)
        return cls._convert_parent(L)
    
    @classmethod
    def from_xr(
            cls,
            R: xr.DataArray,
            pop_dim: str | None = None,
            regime_dim: str | None = None
            ) -> 'NetRegimeWCList':
        """Creates from 2D xarray. """
        L = NetRegime1DList.from_xr(R, pop_dim, regime_dim)
        return cls._convert_parent(L)
