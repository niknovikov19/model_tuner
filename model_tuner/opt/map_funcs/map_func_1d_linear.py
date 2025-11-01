from typing import Dict, List, Tuple

import numpy as np

from .map_func_1d import MapFunc1D, MapFitParams
from .map_func_1d import _is_scalar, _to_scalar, _to_array
from .map_func_1d import _clip_to_nan, _get_empty_bounds_dict


class MapFunc1DLinear(MapFunc1D):
    def __init__(
            self,
            x_limits: Tuple[float, float] = (-np.inf, np.inf),
            y_limits: Tuple[float, float] = (-np.inf, np.inf),
            base_point_num: int | None = None,
            require_increase: bool = True
            ):
        super().__init__()
        self._x_limits = x_limits
        self._y_limits = y_limits
        self.base_point_num = base_point_num
        self.require_increase = require_increase
    
    @classmethod
    def get_par_names(cls) -> List[str]:
        return ['c', 'k']
    
    def f(self, x: float | np.ndarray, c, k) -> float | np.ndarray:
        x_ = _to_array(x)
        x_ = _clip_to_nan(x_, self._x_limits)
        y = k * x_ + c
        y = y.clip(*self._y_limits)
        if _is_scalar(x):
            y = _to_scalar(y)
        return y
    
    def f_inv(self, y: float | np.ndarray, c, k) -> float | np.ndarray:
        y_ = _to_array(y)
        y_ = _clip_to_nan(y_, self._y_limits, need_copy=True)
        x = (y_ - c) / k
        #x = x.clip(*self._x_limits)
        x = _clip_to_nan(x, self._x_limits, need_copy=False)
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
        if self.require_increase and not np.all(np.diff(self.yy) > 0):
            raise ValueError('yy values should be strictly increasing')
        n = self.base_point_num or int(len(xx) / 2)
        k, _ = np.polyfit(self.xx, self.yy, 1)
        c = yy[n] - xx[n] * k
        self.par = {'c': c, 'k': k}
