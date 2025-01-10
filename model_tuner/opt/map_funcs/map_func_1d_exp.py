from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


class MapFunc1DExp(MapFunc1D):    
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'k']

    @staticmethod
    def f(x: Union[float, np.ndarray],
          a, b, k
          ) -> Union[float, np.ndarray]:
        y = a * np.exp(b * x) + k
        #y[y < 0] = np.nan
        return y
    
    @staticmethod
    def f_inv(y: Union[float, np.ndarray],
              a, b, k
              ) -> Union[float, np.ndarray]:
        x = np.log((y - k) / a) / b
        #x[x < 0] = np.nan
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        return [0, 0, -100], [np.inf, np.inf, np.inf]
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        b0 = 1.0
        k0 = np.nanmin(yy) - 0.1 * np.abs(np.nanmin(yy))
        a0 = (np.nanmax(yy) - k0) / np.exp(b0 * np.nanmax(xx))
        return a0, b0, k0