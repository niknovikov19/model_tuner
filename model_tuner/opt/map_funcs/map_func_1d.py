from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit


def _is_1d_array(x: np.ndarray) -> bool:
    return len(x) == len(x.ravel())

def _is_scalar(x: float | np.ndarray) -> bool:
    return not isinstance(x, np.ndarray)

def _to_array(x: float | np.ndarray) -> np.ndarray:
    if _is_scalar(x):
        x = np.array([x])
    return x

def _to_scalar(x: np.ndarray) -> float:
    return x.ravel()[0]

def _clip_to_nan(
        x: np.ndarray,
        limits: Tuple = (None, None),
        need_copy: bool = True
        ) -> np.ndarray:
    x1 = limits[0] or -np.inf
    x2 = limits[1] or np.inf
    mask = (x < x1) | (x > x2)
    y = x.copy() if need_copy else x
    y[mask] = np.nan
    return y

def _get_empty_bounds_dict(
        par_names: List[str]
        ) -> Dict[str, Tuple[float, float]]:
    return {name: (-np.inf, np.inf) for name in par_names}

def _bounds_dict_to_tuple(
        bounds: Dict[str, Tuple[float, float]]
        ) -> Tuple[List[float], List[float]]:
    par_names = list(bounds.keys())
    low = [bounds[name][0] for name in par_names]
    high = [bounds[name][1] for name in par_names]
    return low, high

def _get_empty_bounds(
        par_names: List[str]
        ) -> Tuple[List[float], List[float]]:
    return _bounds_dict_to_tuple(_get_empty_bounds_dict(par_names))

def _mix_tuples(x: Tuple, y: Tuple, alpha: float) -> Tuple:
    z = []
    for xx, yy in zip(x, y):
        z.append((1 - alpha) * xx + alpha * yy)
    return tuple(z)


@dataclass
class MapFitParams:
    ftol: float = 1e-3  
    xtol: float | None = 1e-4  
    verbose: int = False
    method: str = 'trf'
    max_nfev: int = 1000
    par0_kprev: float = 0  # proportion of the previous fit in the initial guess
    return_first_guess: bool = False  # don't do the fitting, return the initial guess
    

class MapFunc1D(ABC):
    
    def __init__(self):
        self.par: Dict[str, float] = {name: np.nan for name in self.get_par_names()}
    
    @classmethod
    @abstractmethod
    def get_par_names(cls) -> List[str]:
        pass

    def get_par_vals(self) -> List:
        return [self.par[name] for name in self.get_par_names()]
        
    @abstractmethod
    def f(self, x: float | np.ndarray, *args, **kwargs) -> float | np.ndarray:
        pass
    
    @abstractmethod
    def f_inv(self, y: float | np.ndarray, *args, **kwargs) -> float | np.ndarray:
        pass
    
    def apply(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.f(x, **self.par)
    
    def apply_inv(self, y: float | np.ndarray) -> float | np.ndarray:
        return self.f_inv(y, **self.par)
    
    @classmethod
    def _get_fit_bounds(cls) -> Dict[str, Tuple[float, float]]:
        return _get_empty_bounds_dict(cls.get_par_names())  # can be overloaded in a subclass
        
    @classmethod
    @abstractmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:
        pass

    def is_valid(self) -> bool:
        return not np.any(np.isnan(list(self.par.values())))
    
    #def mix_params(self, par_mix: Dict[str, float], alpha: float) -> None:
    #    for par_name, val in self.par:
    #        self.par[par_name] = (1 - alpha) * val + alpha * par_mix[par_name]
    
    def fit(
            self,
            xx: np.ndarray,
            yy: np.ndarray,
            opt_par: MapFitParams = MapFitParams(),
            ww: np.ndarray | None = None,
            bounds: Dict[str, Tuple[float, float]] | None = None
            ) -> None:
        
        # Convert data to 1-d format
        if not _is_1d_array(xx) or not _is_1d_array(yy):
            raise ValueError('xx and yy should be effectively 1-dimensional')
        xx, yy = xx.ravel(), yy.ravel()
        if ww is not None:
            if not _is_1d_array(ww):
                raise ValueError('ww should be effectively 1-dimensional')
            ww = ww.ravel()
        
        # Initial guess
        par0 = self._get_first_fit_guess(xx, yy)  # default first guess (from a subclass)
        par0_prev = self.get_par_vals()  # result of the previous fitting
        if not np.any(np.isnan(par0_prev)):
            par0 = _mix_tuples(par0, par0_prev, opt_par.par0_kprev)  # mix par0 and par0_prev
        
        # Accept the initial guess without further fitting
        if opt_par.return_first_guess:
            self.par = {
                name: par0[n] for n, name in enumerate(self.get_par_names())
            }
            return
        
        # Bounds of fitting
        bounds = bounds or {}
        bounds_def = self._get_fit_bounds()  # default bounds (from a subclass)
        bounds = bounds_def | bounds  # replace bounds provided in arguments
        bounds = _bounds_dict_to_tuple(bounds)

        # Fit
        try:
            par, _ = curve_fit(
                self.f, xx, yy, p0=par0, bounds=bounds, nan_policy='omit',
                sigma=ww, absolute_sigma=(ww is not None),
                ftol=opt_par.ftol, xtol=opt_par.xtol,
                method=opt_par.method, max_nfev=opt_par.max_nfev,
                verbose=opt_par.verbose
            )
            self.par = {
                name: par[n] for n, name in enumerate(self.get_par_names())
            }
        except Exception as e:
            print(f'Fitting failed ({e})')
            self.par = {name: np.nan for name in self.get_par_names()}
