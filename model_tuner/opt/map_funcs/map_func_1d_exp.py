from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


class MapFunc1DExp(MapFunc1D):
    def __init__(
            self,
            x_positive: bool = False,
            y_positive: bool = False
            ):
        self._x_positive = x_positive
        self._y_positive = y_positive
    
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'k']

    #@staticmethod
    def f(self,
          x: float | np.ndarray,
          a, b, k
          ) -> float | np.ndarray:
        
        is_scalar = not isinstance(x, np.ndarray)
        if is_scalar:
            x = np.array([x])
        
        if self._x_positive:
            x[x < 0] = np.nan
            
        y = a * np.exp(b * x) + k
        
        if self._y_positive:
            y = np.maximum(0, y)
            
        if is_scalar:
            y = y.ravel()[0]
            
        return y
    
    #@staticmethod
    def f_inv(self,
              y: float | np.ndarray,
              a, b, k
              ) -> float | np.ndarray:
        
        is_scalar = not isinstance(y, np.ndarray)
        if is_scalar:
            y = np.array([y])
            
        if self._y_positive:
            y[y < 0] = np.nan
            
        y[y < k] = np.nan
        x = np.log((y - k) / a) / b
        
        if self._x_positive:
            x = np.maximum(0, x)
        
        if is_scalar:
            x = x.ravel()[0]
            
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        return [0, 0, -100], [np.inf, np.inf, np.inf]
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        b00 = 10.0
        b0 = b00 / np.nanmax(xx)
        k0 = np.nanmin(yy) - 0.1 * np.abs(np.nanmin(yy))
        a0 = (np.nanmax(yy) - k0) / np.exp(b00)
        return a0, b0, k0