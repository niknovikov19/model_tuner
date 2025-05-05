from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D
from .map_func_1d import _is_scalar, _to_scalar, _to_array, _clip_to_nan


class MapFunc1DRational21(MapFunc1D):
    """Function: y = (ax^2 + bx + c) / (x + d) """
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
        return ['a', 'b', 'c', 'd']
    
    def f(self, x: float | np.ndarray, a, b, c, d) -> float | np.ndarray:
        x_ = _to_array(x)
        x_ = _clip_to_nan(x_, self._x_limits)    
        y = (a * x**2 + b * x + c) / (x + d)
        y = y.clip(*self._y_limits)
        if _is_scalar(x):
            y = _to_scalar(y)
        return y
    
    def f_inv(self, y: float | np.ndarray, a, b, c, d) -> float | np.ndarray:
        y_ = _to_array(y)
        y_ = _clip_to_nan(y_, self._y_limits, need_copy=True)

        B = b - y_
        C = c - y_ * d

        # Quadratic case
        if not np.isclose(a, 0):
            disc = B**2 - 4 * a * C
            y_[disc < 0] = np.nan   # set non-invertible points to nan
            disc[disc < 0] = np.nan            
            sqrt_disc = np.sqrt(disc)
            x_plus  = (y_ - b + sqrt_disc) / (2 * a)
            x_minus = (y_ - b - sqrt_disc) / (2 * a)
            x = max(x_plus, x_minus)
        # Linear case
        else:
            y_[np.isclose(B, 0)] = np.nan   # set non-invertible points to nan
            x = (y_ * d - c) / B

        x = x.clip(*self._x_limits)
        if _is_scalar(y):
            x = _to_scalar(x)
        return x
    
    @classmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:

        # y = (ax^2 + bx + c) / (x + d) 

        # Fit a * x + b to the data
        a0, b0 = np.polyfit(xx, yy, 1)

        # Pick -d0 well to the left of xx[0]
        #d0 = -(xx[0] - 1.5 * (xx[1] - xx[0]))
        d0 = 20

        # Calculate c0 to minimize the total error
        #w = 1.0 / (xx + d0)
        #g = (a0 * xx**2 + b0 * xx) * w
        #c0 = np.dot(w, (yy - g)) / np.dot(w, w)
        c0 = -5

        return a0, b0, c0, d0
