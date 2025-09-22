from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

from .map_func_1d import MapFunc1D, MapFitParams
from .map_func_1d import _is_scalar, _to_scalar, _to_array
from .map_func_1d import _clip_to_nan, _get_empty_bounds_dict


def _interpolate(x, xx, yy, spline_: PPoly):
    x = np.asarray(x)
    y = np.zeros_like(x)
    mask_l = x < xx[0]
    mask_r = x > xx[-1]
    mask_m = (xx[0] <= x) & (x <= xx[-1])
    y[mask_l] = yy[0] + spline_(xx[0], 1) * (x[mask_l] - xx[0])
    y[mask_r] = yy[-1] + spline_(xx[-1], 1) * (x[mask_r] - xx[-1])
    y[mask_m] = spline_(x[mask_m])
    if np.isscalar(x):
        y = y[0]
    return y

def _interpolate_inv(y, xx, yy, spline_: PPoly):
    y = np.asarray(y)
    x = np.zeros_like(y)
    for n, y_ in enumerate(y):
        if y_ < yy[0]:
            x[n] = xx[0] + (y_ - yy[0]) / spline_(xx[0], 1)
        elif y_ > yy[-1]:
            x[n] = xx[-1] + (y_ - yy[-1]) / spline_(xx[-1], 1)
        else:
            x[n] = spline_.solve(y_, extrapolate=False)[0]
    if np.isscalar(y):
        x = x[0]
    return x


class MapFunc1DSpline(MapFunc1D):
    def __init__(
            self,
            x_limits: Tuple[float, float] = (-np.inf, np.inf),
            y_limits: Tuple[float, float] = (-np.inf, np.inf),
            spline_type: str | None = None
            ):
        super().__init__()
        self._x_limits = x_limits
        self._y_limits = y_limits
        self.xx = None
        self.yy = None
        self.spline = None
        self.spline_type = spline_type or 'cubic'
    
    @classmethod
    def get_par_names(cls) -> List[str]:
        return []
    
    def f(self, x: float | np.ndarray) -> float | np.ndarray:
        x_ = _to_array(x)
        x_ = _clip_to_nan(x_, self._x_limits)
        y = _interpolate(x_, self.xx, self.yy, self.spline)
        y = y.clip(*self._y_limits)
        if _is_scalar(x):
            y = _to_scalar(y)
        return y
    
    def f_inv(self, y: float | np.ndarray) -> float | np.ndarray:
        y_ = _to_array(y)
        y_ = _clip_to_nan(y_, self._y_limits, need_copy=True)
        x = _interpolate_inv(y_, self.xx, self.yy, self.spline)
        x = x.clip(*self._x_limits)
        if _is_scalar(y):
            x = _to_scalar(x)
        return x

    @classmethod
    def _get_fit_bounds(cls) -> Dict[str, Tuple[float, float]]:
        bounds = _get_empty_bounds_dict(cls.get_par_names())
        return bounds
    
    @classmethod
    def _get_first_fit_guess(
            cls, xx: np.ndarray, yy: np.ndarray,
            bounds: Dict[str, Tuple[float, float]]
            ) -> Tuple:
        return tuple()
    
    def fit(
            self,
            xx: np.ndarray,
            yy: np.ndarray,
            opt_par: MapFitParams = MapFitParams(),
            ww: np.ndarray | None = None,
            bounds: Dict[str, Tuple[float, float]] | None = None
            ) -> None:
        idx = np.argsort(xx)
        self.xx, self.yy = xx[idx], yy[idx]
        if self.spline_type == 'cubic':
            self.spline = CubicSpline(self.xx, self.yy, extrapolate=False)
        elif self.spline_type == 'pchip':
            self.spline = PchipInterpolator(self.xx, self.yy, extrapolate=False)
        else:
            raise ValueError(f'Unsupported spline_type: {self.spline_type}')
