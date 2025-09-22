from typing import Dict, List, Tuple, Union

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
    def _get_first_fit_guess(
            cls, xx: np.ndarray, yy: np.ndarray,
            bounds: Dict[str, Tuple[float, float]]
            ) -> Tuple:
        """
        First guess for:
            y = y0 + m * (x - a) / (1 + exp[-k (x - x0)])

        Robust for as few as 4 points:
        1) y0 from left block (median of first 2).
        2) Right-tail line from last 2 points: secant slope m, intercept c_fit.
        3) a so that right asymptote matches that line: c_fit = y0 - m*a.
        4) Fit k, x0 by linear regression of logit(r_i) vs x_i where
            r_i = (y_i - y0) / (m * (x_i - a))  clipped to (eps, 1-eps).
        5) Enforce bounds:
            - If y0 is clipped → a += Δ/m, then refit k,x0.
            - If m is clipped  → a = (y0 - c_fit)/m, then refit k,x0.
            - Finally clip k, x0, a (no further compensation).
        """
        # ---- sanitize & sort ----
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        if xx.ndim != 1 or yy.ndim != 1 or len(xx) != len(yy):
            raise ValueError("xx and yy must be 1D arrays of equal length")
        n = len(xx)
        if n < 4:
            raise ValueError("Need at least 4 points")

        o = np.argsort(xx)
        xx, yy = xx[o], yy[o]

        # ---- 1) left baseline y0 (median of first 2) ----
        y0_raw = float(np.median(yy[:min(2, n)]))

        # ---- 2) right-tail line from last two points ----
        x1, x2 = xx[-2], xx[-1]
        y1, y2 = yy[-2], yy[-1]
        dx = max(1e-12, x2 - x1)
        m_fit = float((y2 - y1) / dx)
        c_fit = float(y2 - m_fit * x2)  # intercept of the secant line y = m x + c_fit

        # ---- 3) anchor 'a' so right asymptote matches the line ----
        m_safe = m_fit if abs(m_fit) > 1e-12 else (1e-12 if m_fit >= 0 else -1e-12)
        a_guess = (y0_raw - c_fit) / m_safe

        # ---- helper: fit k, x0 from fractional rise r = (y - y0)/(m(x-a)) ----
        def fit_k_x0(y0_val: float, m_val: float, a_val: float) -> Tuple[float, float]:
            denom = m_val * (xx - a_val)
            # use points with denom>0 (on the rising side)
            mask = denom > 0
            if mask.sum() < 2:
                # fallback: use all, but avoid zeros/negatives by shifting a slightly
                a_val = float(a_val - 1e-6)
                denom = m_val * (xx - a_val)
                mask = denom > 0

            r = (yy[mask] - y0_val) / denom[mask]
            # clip r to (eps, 1-eps) to avoid inf logits
            eps = 1e-6
            r = np.clip(r, eps, 1 - eps)
            z = np.log(r) - np.log1p(-r)  # logit

            x_sel = xx[mask]
            # linear regression: z ≈ k*x - k*x0
            A = np.vstack([x_sel, np.ones_like(x_sel)]).T
            k_est, b_est = np.linalg.lstsq(A, z, rcond=None)[0]
            # if k too small or nan, nudge
            if not np.isfinite(k_est) or abs(k_est) < 1e-12:
                k_est = 1.0  # mild default
            x0_est = -b_est / k_est
            return float(k_est), float(x0_est)

        # initial k,x0 from raw y0/m/a
        k_guess, x0_guess = fit_k_x0(y0_raw, m_fit, a_guess)

        # ---- staged bound projection with compensations & refits ----
        def clip_if_present(val: float, name: str) -> float:
            if name in bounds:
                lo, hi = bounds[name]
                return float(np.clip(val, lo, hi))
            return float(val)

        m, k, x0, a, y0 = float(m_fit), float(k_guess), float(x0_guess), float(a_guess), float(y0_raw)

        # clip y0, compensate 'a', then refit k,x0
        if 'y0' in bounds:
            y0_clipped = clip_if_present(y0, 'y0')
            if y0_clipped != y0:
                a += (y0_clipped - y0) / (m if abs(m) > 1e-12 else (1e-12 if m >= 0 else -1e-12))
                y0 = y0_clipped
                k, x0 = fit_k_x0(y0, m, a)

        # clip m, compensate 'a', then refit k,x0
        if 'm' in bounds:
            m_clipped = clip_if_present(m, 'm')
            if m_clipped != m:
                m = m_clipped if abs(m_clipped) > 1e-12 else (1e-12 if m_clipped >= 0 else -1e-12)
                a = (y0 - c_fit) / m
                k, x0 = fit_k_x0(y0, m, a)

        # finally clip k, x0, a
        k  = clip_if_present(k,  'k')
        x0 = clip_if_present(x0, 'x0')
        a  = clip_if_present(a,  'a')

        return m, k, x0, a, y0



        """ # --- sanitise / sort -------------------------------------------------
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]
        n = len(xx)
        frac_plateau = 0.1
        n_end = max(3, int(frac_plateau * n))

        # --- 1. left horizontal level  ---------------------------------------
        y0 = float(np.median(yy[:n_end]))
        if 'y0' in bounds:
            y0 = np.clip(y0, *bounds['y0'])

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
    """