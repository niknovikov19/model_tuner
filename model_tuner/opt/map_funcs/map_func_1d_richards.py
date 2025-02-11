from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


class MapFunc1DRichards(MapFunc1D):
    
    @classmethod
    def get_par_names(cls) -> List[str]:
        return ['a', 'b', 'c', 'k', 'q']
    
    def f(self, x: np.ndarray, a, b, c, k, q) -> np.ndarray:
        y = c + a / ((1 + np.exp(-k * (x - b))) ** q)
        return y
    
    def f_inv(self, y: np.ndarray, a, b, c, k, q) -> np.ndarray:
        y = y.copy()
        y[(a / (y - c)) < 1] = np.nan
        x = b - np.log((a / (y - c)) ** (1/q) - 1) / k
        return x
    
    @classmethod
    def _get_fit_bounds(cls) -> Tuple[List[float], List[float]]:
        par_names = cls.get_par_names()
        bounds = {p: (-np.inf, np.inf) for p in par_names}
        bounds['q'] = (1, 5)
        low = [bounds[p][0] for p in par_names]
        high = [bounds[p][1] for p in par_names]
        return low, high
    
    @classmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:
        is_increasing = yy[-1] > yy[0]

        c0 = np.nanmin(yy)
        a0 = np.nanmax(yy) - c0        
        q0 = 1
    
        # Points closest to the middle threshold 
        th = c0 + a0 / 2
        mask1 = yy < th
        mask2 = yy > th
        n1 = np.argmax(yy[mask1])  # Highest value below threshold
        n2 = np.argmin(yy[mask2])  # Lowest value above threshold
        x1, y1 = xx[mask1][n1], yy[mask1][n1]
        x2, y2 = xx[mask2][n2], yy[mask2][n2]
        
        # Compute b0 as the midpoint of transition
        b0 = (x1 + x2) / 2
    
        # Adjust k0 based on monotonicity
        #k0 = 1 / np.max(np.abs(xx))
        #if not is_increasing:
        #    k0 = -k0  # Flip sign if decreasing
        k0 = 4 * (y2 - y1) / (x2 - x1) / a0
        
        return a0, b0, c0, k0, q0