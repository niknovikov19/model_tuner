from typing import Dict, List, Tuple

import numpy as np

from .map_func_1d import MapFunc1D
from .map_func_1d import _is_scalar, _to_scalar, _to_array
from .map_func_1d import _clip_to_nan, _get_empty_bounds_dict


class MapFunc1DRichards(MapFunc1D):
    def __init__(
            self,
            x_limits: Tuple[float, float] = (-np.inf, np.inf),
            y_limits: Tuple[float, float] = (-np.inf, np.inf)
            ):
        super().__init__()
        self._x_limits = x_limits
        self._y_limits = y_limits
    
    @classmethod
    def get_par_names(cls) -> List[str]:
        return ['a', 'b', 'c', 'k', 'q']
    
    def f(self, x: float | np.ndarray, a, b, c, k, q) -> float | np.ndarray:
        x_ = _to_array(x)
        x_ = _clip_to_nan(x_, self._x_limits)    
        y = c + a / ((1 + np.exp(-k * (x_ - b))) ** q)
        y = y.clip(*self._y_limits)
        if _is_scalar(x):
            y = _to_scalar(y)
        return y
    
    def f_inv(self, y: float | np.ndarray, a, b, c, k, q) -> float | np.ndarray:
        y_ = _to_array(y)
        y_ = _clip_to_nan(y_, self._y_limits, need_copy=True)
        y_[(a / (y_ - c)) < 0] = np.nan
        y_[(a / (y_ - c)) ** (1/q) < 1] = np.nan
        x = b - np.log((a / (y_ - c)) ** (1/q) - 1) / k
        x = x.clip(*self._x_limits)
        if _is_scalar(y):
            x = _to_scalar(x)
        return x

    @classmethod
    def _get_fit_bounds(cls) -> Dict[str, Tuple[float, float]]:
        bounds = _get_empty_bounds_dict(cls.get_par_names())
        bounds['q'] = (1, 5)
        return bounds
    
    @classmethod
    def _get_first_fit_guess(
            cls, xx: np.ndarray, yy: np.ndarray,
            bounds: Dict[str, Tuple[float, float]]
            ) -> Tuple:

        q0 = 1

        c0 = np.nanmin(yy)
        if 'c' in bounds:
            c0 = np.clip(c0, *bounds['c'])

        a0 = np.nanmax(yy) - c0
        
        # Interval with the max. slope and its midpoint
        dy = yy[1:] - yy[:-1]
        n = np.nanargmax(dy)
        x01, y01 = xx[n], yy[n]
        x02, y02 = xx[n + 1], yy[n + 1]
        b0 = (x01 + x02) / 2
        
        # Corner points
        x1, y1 = np.nanmin(xx), np.nanmin(yy)
        x2, y2 = np.nanmax(xx), np.nanmax(yy)
        
        for _ in range(5):
            k0 = 4 * (y02 - y01) / (x02 - x01) / a0
            S = lambda x: 1 / (1 + np.exp(-k0 * (x - b0)))
            a0 = (y2 - y1) / (S(x2) - S(x1))
            c0 = y1 - a0 * S(x1)
        
        return a0, b0, c0, k0, q0