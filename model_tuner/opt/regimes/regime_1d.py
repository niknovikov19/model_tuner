from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from model_tuner.utils import copy_or_ref
from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegime1D(PopRegime):
    value: float = 0
    
    def is_valid(self) -> bool:
        return not np.isnan(self.value)
    
    def mix_with(self, R: 'PopRegime1D', alpha) -> None:
        self.value = (1 - alpha) * self.value + alpha * R.value
    
    def copy(self) -> 'PopRegime1D':
        return type(self)(self.value)
    
    @classmethod
    def create(
            cls,
            R: Union[float, 'PopRegime1D']
            ) -> 'PopRegime1D':  
        if isinstance(R, float):
            return cls(value=R)
        elif isinstance(R, PopRegime1D):
            return cls(value=R.value)
        else:
            raise TypeError('R should be a float or PopRegime1D object')


class NetRegime1D(NetRegime):
    def __init__(
            self,
            R: (NetRegime | Dict[str, float | PopRegime1D] | 
                List[float | PopRegime1D] | None) = None,
            pop_names: List[str] | None = None,
            force_copy: bool = False
            ):
        # Empty     
        if R is None:
            self.pop_regimes = {}
        # From NetRegime with PopRegime1D entries
        elif isinstance(R, NetRegime):
            self.pop_regimes = copy_or_ref(R.pop_regimes, force_copy)
        # From a list of float values or PopRegime1D objects
        elif isinstance(R, list):
            if pop_names is None:
                pop_names = [f'pop{n}' for n in range(len(R))]
            self._from_values(pop_names, R)
        # From a dict with float values or PopRegime1D objects
        elif isinstance(R, dict):
            self._from_dict(R)
        
        self._check()

    def _check(self) -> None:
        for R_ in self.pop_regimes.values():
            if not isinstance(R_, PopRegime1D):
                raise TypeError(
                    'NetRegime1D should contain PopRegime1D objects.'
                )
        super()._check()

    def get_pop_regime_val(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].value
    
    def get_pop_regimes_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('value')
    
    def _from_values(
            self,
            pop_names: List[str],
            pop_values: List[float | PopRegime1D]
            ) -> None:
        self.pop_regimes = {}
        for pop_name, val in zip(pop_names, pop_values):
            if isinstance(val, PopRegime1D):
                self.pop_regimes[pop_name] = val.copy()
            else:
                self.pop_regimes[pop_name] = PopRegime1D(value=val)
        self._check()
    
    @classmethod
    def from_values(
            cls,
            pop_names: List[str],
            pop_values: List[float | PopRegime1D]
            ) -> 'NetRegime1D':
        R = cls()
        R._from_values(pop_names, pop_values)
        return R

    def _from_dict(
            self,
            pop_vals_dict: Dict[str, float | PopRegime1D]
            ) -> None:
        self._from_values(
            pop_names=list(pop_vals_dict.keys()),
            pop_values=list(pop_vals_dict.values())
        )
    
    @classmethod
    def from_dict(
            cls,
            pop_vals_dict: Dict[str, float | PopRegime1D]
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


class NetRegime1DList(NetRegimeList):
    def __init__(
            self,
            L: (NetRegimeList | List[NetRegime] |
                np.ndarray | xr.DataArray | None) = None,
            pop_names: List[str] | None = None,
            force_copy: bool = False,
            ):
        # Empty
        if L is None:
            self.net_regimes = []
        # From NetRegimeList
        elif isinstance(L, NetRegimeList):
            self.net_regimes = []
            for R in L.net_regimes:
                T = type(R) if isinstance(R, NetRegime1D) else NetRegime1D
                self.net_regimes.append(T(R, force_copy=force_copy))
        # From a list of NetRegime objects
        elif isinstance(L, list):
            self.net_regimes = []
            for R in L:
                T = type(R) if isinstance(R, NetRegime1D) else NetRegime1D
                self.net_regimes.append(T(R, force_copy=force_copy))
        # From (pops x points) ndarray
        elif isinstance(L, np.ndarray):
            if pop_names is None:
                pop_names = [f'pop{n}' for n in range(L.shape[1])]
            self._from_regimes_mat(pop_names, L)
        # From (pops x points) xarray
        elif isinstance(L, xr.DataArray):
            self._from_xr(L)
        
        self._check()
    
    def _check(self):
        for R in self.net_regimes:
            if not isinstance(R, NetRegime1D):
                raise TypeError(
                    'NetRegime1DList should contain NetRegime1D objects.'
                )
        super()._check()
    
    def get_pop_regimes_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('value')
    
    def to_xr(
            self,
            dim_name: str = 'regime',
            labels: List | None = None
            ) -> xr.DataArray:
        """Returns (pops x regimes) xarray. """
        if labels is None:
            labels = np.arange(len(self))
        return xr.DataArray(
            self.get_pop_regimes_mat(),
            dims=['pop', dim_name],
            coords={'pop': self.get_pop_names(), dim_name: labels}
        )
    
    def _from_regimes_mat(
            self,
            pop_names: List[str],
            regimes_mat: np.ndarray  # pops x regimes
            ) -> None:
        """Creates from (pops x regimes) matrix. """
        if regimes_mat.ndim != 2:
                raise ValueError('regimes_mat should be 2D')
        self.net_regimes = []
        for n in range(regimes_mat.shape[1]):
            regime_vals = regimes_mat[:, n]
            self.net_regimes.append(
                NetRegime1D.from_values(pop_names, regime_vals)
            )
    
    @classmethod
    def from_regimes_mat(
            cls,
            pop_names: List[str],
            regimes_mat: np.ndarray  # pops x regimes
            ) -> 'NetRegime1DList':
        """Creates from (pops x regimes) matrix. """
        L = cls()
        L._from_regimes_mat(pop_names, regimes_mat)
        return L
    
    def _from_xr(
            self,
            R: xr.DataArray,
            pop_dim: str | None = None,
            regime_dim: str | None = None
            ) -> None:
        """Creates from 2D xarray. """
        if R.ndim != 2:
            raise ValueError('R should be 2D')
        if pop_dim and pop_dim not in R.dims:
            raise ValueError(f'{pop_dim} not in R.dims')
        if regime_dim and regime_dim not in R.dims:
            raise ValueError(f'{regime_dim} not in R.dims')
        if not pop_dim and not regime_dim:
            if 'pop' in R.dims:
                pop_dim = 'pop'
            if 'regime' in R.dims:
                regime_dim = 'regime'
        if not pop_dim and not regime_dim:
            raise ValueError('Cannot interpret R.dims')
        if not regime_dim and pop_dim in R.dims:
            regime_dim = [d for d in R.dims if d != pop_dim][0]
        if not pop_dim and regime_dim in R.dims:
            pop_dim = [d for d in R.dims if d != regime_dim][0]
        if pop_dim == R.dims[0]:
            R_ = R.values
        else:
            R_ = R.values.T
        self._from_regimes_mat(
            pop_names=R.coords[pop_dim].values,
            regimes_mat=R_
        )
    
    @classmethod
    def from_xr(
            cls,
            R: xr.DataArray,
            pop_dim: str | None = None,
            regime_dim: str | None = None
            ) -> 'NetRegime1DList':
        """Creates from 2D xarray. """
        L = cls()
        L._from_xr(R, pop_dim, regime_dim)
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
