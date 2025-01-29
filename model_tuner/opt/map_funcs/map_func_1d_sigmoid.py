from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


class MapFunc1DSigmoid(MapFunc1D):
    def __init__(
            self,
            x_positive: bool = False,
            y_positive: bool = False
            ):
        self._x_positive = x_positive
        self._y_positive = y_positive
    
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'c', 'k']
    
    def f(self,
          x: float | np.ndarray,
          a, b, c, k
          ) -> float | np.ndarray:
        
        is_scalar = not isinstance(x, np.ndarray)
        if is_scalar:
            x = np.array([x])
        
        if self._x_positive:
            x[x < 0] = np.nan
            
        y = c + a / (1 + np.exp(-k * (x - b)))
        
        if self._y_positive:
            y = np.maximum(0, y)
            
        if is_scalar:
            y = y.ravel()[0]

        return y
    
    def f_inv(self,
              y: float | np.ndarray,
              a, b, c, k
              ) -> float | np.ndarray:
        
        is_scalar = not isinstance(y, np.ndarray)
        if is_scalar:
            y = np.array([y])
            
        if self._y_positive:
            y[y < 0] = np.nan
            
        y[(a / (y - c)) < 1] = np.nan
        x = b - np.log(a / (y - c) - 1) / k
        
        if self._x_positive:
            x = np.maximum(0, x)
        
        if is_scalar:
            x = x.ravel()[0]
        
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        bounds = {p: (-np.inf, np.inf) for p in ['a', 'b', 'c', 'k']}
# =============================================================================
#         bounds = {
#             'a': (0.1, 10),
#             'b': (-10, 20),
#             'c': (-10, 20),
#             'k': (0.1, 10)
#         }
# =============================================================================
        low = [bounds[p][0] for p in ['a', 'b', 'c', 'k']]
        high = [bounds[p][1] for p in ['a', 'b', 'c', 'k']]
        return low, high
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        k0 = 1 / np.max(np.abs(xx))
        c0 = np.nanmin(yy)
        a0 = np.nanmax(yy) - c0
        th = c0 + a0 / 2
        mask1 = yy < th
        mask2 = yy > th
        n1 = np.argmax(yy[mask1])
        n2 = np.argmin(yy[mask2])
        x1, y1 = xx[mask1][n1], yy[mask1][n1]
        x2, y2 = xx[mask2][n2], yy[mask2][n2]
        b0 = (x1 + x2) / 2
        return a0, b0, c0, k0