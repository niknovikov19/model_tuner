from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegime1D(PopRegime):
    value: float = 0
    
    def is_valid(self) -> bool:
        return not np.isnan(self.value)
    
    def mix_with(self, R: 'PopRegime1D', alpha) -> None:
        self.value = (1 - alpha) * self.value + alpha * R.value
    
    def copy(self) -> 'PopRegime1D':
        return PopRegime1D(self.value)


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
            pop_values: List[float | PopRegime1D]
            ) -> 'NetRegime1D':
        R = NetRegime1D()
        for pop_name, val in zip(pop_names, pop_values):
            if isinstance(val, PopRegime1D):
                R.pop_regimes[pop_name] = val.copy()
            else:
                R.pop_regimes[pop_name] = PopRegime1D(value=val)
        return R
    
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


@dataclass
class NetRegime1DList(NetRegimeList):
    
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
    
    @classmethod
    def from_regimes_mat(
            cls,
            pop_names: List[str],
            regimes_mat: np.ndarray  # pops x regimes
            ) -> 'NetRegime1DList':
        """Creates from (pops x regimes) matrix. """
        L = NetRegime1DList()
        for n in range(regimes_mat.shape[1]):
            regime_vals = regimes_mat[:, n]
            L.net_regimes.append(
                NetRegime1D.from_values(pop_names, regime_vals)
            )
        return L
    
    @classmethod
    def from_xr(
            cls,
            R: xr.DataArray,
            pop_dim: str | None = None,
            regime_dim: str | None = None
            ) -> 'NetRegime1DList':
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
        return cls.from_regimes_mat(
            pop_names=R.coords[pop_dim].values,
            regimes_mat=R_
        )
    
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
        
    
