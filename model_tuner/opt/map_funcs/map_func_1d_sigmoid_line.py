from typing import List, Tuple, Union

import numpy as np
from scipy.special import lambertw

from .map_func_1d import MapFunc1D
from .map_func_1d import _is_scalar, _to_scalar, _to_array, _clip_to_nan


class MapFunc1DSigmoidLine(MapFunc1D):
    """Function: y = y0 + m * (x - a) / (1 + exp[-k(x - x0)]) """
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
        return ['m', 'k', 'x0', 'a', 'y0']
    
    def f(self, x: float | np.ndarray, m, k, x0, a, y0) -> float | np.ndarray:
        x_ = _to_array(x)
        x_ = _clip_to_nan(x_, self._x_limits)

        y = y0 + m * (x_ - a) / (1 + np.exp(-k * (x_ - x0)))

        y = y.clip(*self._y_limits)
        if _is_scalar(x):
            y = _to_scalar(y)
        return y
    
    def f_inv(self, y: float | np.ndarray, m, k, x0, a, y0) -> float | np.ndarray:
        y_ = _to_array(y)
        y_ = _clip_to_nan(y_, self._y_limits, need_copy=True)

        B = (y_ - y0) / m         # normalized output
        C = x0 - a
        z = k * B * np.exp(-k * (B - C))
        w = lambertw(z)           # principal branch (real because z>0 here)
        x = a + B + (w.real / k)

        x = x.clip(*self._x_limits)
        if _is_scalar(y):
            x = _to_scalar(x)
        return x
    
    @classmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:

        # --- sanitise / sort -------------------------------------------------
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]
        n = len(xx)
        frac_plateau = 0.1
        n_end = max(3, int(frac_plateau * n))

        # --- 1. left horizontal level  ---------------------------------------
        y0 = float(np.median(yy[:n_end]))

        # --- 2. right linear tail (slope m and intercept) --------------------
        xr, yr = xx[-n_end:], yy[-n_end:]
        A = np.vstack([xr, np.ones_like(xr)]).T
        m, c = np.linalg.lstsq(A, yr, rcond=None)[0]

        # anchor a so that the right tail passes through the data
        a = (y0 - c) / m

        # --- 3. logistic midpoint x0 ----------------------------------------
        y_right = float(np.median(yr))
        y_mid   = 0.5 * (y0 + y_right)
        # x where yy crosses y_mid (linear interpolation)
        x0 = float(np.interp(y_mid, yy, xx))

        # --- 4. steepness k from 10–90 % width ------------------------------
        y10 = y0 + 0.10 * (y_right - y0)
        y90 = y0 + 0.90 * (y_right - y0)
        x10 = float(np.interp(y10, yy, xx))
        x90 = float(np.interp(y90, yy, xx))
        width = max(1e-12, x90 - x10)          # avoid divide‑by‑zero
        k = 4.394 / width                      # 4.394 ≈ ln(9)*2

        return m, k, x0, a, y0
