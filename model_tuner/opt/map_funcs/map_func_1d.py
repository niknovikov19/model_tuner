from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit


def _is_1d_array(x: np.ndarray) -> bool:
    return len(x) == len(x.ravel())


class MapFunc1D(ABC):
    
    def __init__(
            self,
            x_positive: bool = False,
            y_positive: bool = False
            ):
        self.par = {name: np.nan for name in self.get_par_names()}
        self._x_positive = x_positive
        self._y_positive = y_positive
    
    @classmethod
    @abstractmethod
    def get_par_names(cls) -> List[str]:
        pass

    def get_par_vals(self) -> List:
        return [self.par[name] for name in self.get_par_names()]
        
    @abstractmethod
    def f(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def f_inv(self, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass
    
    def _apply(
            self, x: float | np.ndarray, *args, **kwargs
            ) -> float | np.ndarray:

        is_scalar = not isinstance(x, np.ndarray)
        if is_scalar:
            x = np.array([x])
        
        if self._x_positive:
            x[x < 0] = np.nan
            
        y = self.f(x, *args, **kwargs)
        
        if self._y_positive:
            y = np.maximum(0, y)
            
        if is_scalar:
            y = y.ravel()[0]

        return y
    
    def _apply_inv(
            self, y: float | np.ndarray, *args, **kwargs
            ) -> float | np.ndarray:
        
        is_scalar = not isinstance(y, np.ndarray)
        if is_scalar:
            y = np.array([y])
        
        if self._y_positive:
            y[y < 0] = np.nan
        
        x = self.f_inv(y, *args, **kwargs)
        
        if self._x_positive:
            x = np.maximum(0, x)
        
        if is_scalar:
            x = x.ravel()[0]
        
        return x
    
    def apply(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._apply(x, **self.par)
    
    def apply_inv(self, y: float | np.ndarray) -> float | np.ndarray:
        return self._apply_inv(y, **self.par)
    
    @classmethod
    @abstractmethod
    def _get_fit_bounds(cls) -> Tuple[List[float], List[float]]:
        pass
        
    @classmethod
    @abstractmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:
        pass

    def is_valid(self) -> bool:
        return not np.any(np.isnan(list(self.par.values())))
    
    def mix(self, fmix: 'MapFunc1D', alpha: float) -> None:
        for par_name, val in self.par:
            self.par[par_name] = (1 - alpha) * val + alpha * fmix.par[par_name]
    
    def fit(self, xx: np.ndarray, yy: np.ndarray, from_prev=False) -> None:
        
        # Convert both arrays to 1-d format
        if not _is_1d_array(xx) or not _is_1d_array(yy):
            raise ValueError('xx and yy should be effectively 1-dimentional')
        xx, yy = xx.ravel(), yy.ravel()
            
        # Initial guess
        if from_prev:
            par0 = self.get_par_vals()  # use the result of the previous fitting
        else:
            par0 = self._get_first_fit_guess(xx, yy)  # defined in subclasses
            
        # Bounds of fitting
        bounds = self._get_fit_bounds()  # defined in subclasses
        
        # Fit
        try:
            sigma = np.clip(yy ** 0.5, 0.1, 5)
            par, _ = curve_fit(
                self._apply, xx, yy, p0=par0, bounds=bounds, nan_policy='omit',
                #sigma=sigma, absolute_sigma=True,
                ftol=1e-3, xtol=1e-4, verbose=0, method='trf', max_nfev=1000
            )  
            #par = par0
            self.par = {
                name: par[n] for n, name in enumerate(self.get_par_names())
            }
        except Exception as e:
            print(f'Fitting failed ({e})')
            self.par = {name: np.nan for name in self.get_par_names()}
