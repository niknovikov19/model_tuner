from matplotlib import pyplot as plt
import numpy as np
from scipy.special import lambertw     # pip install mpmath  (or use scipy.special.lambertw)

# ---------------------------------------------------------------------
# Forward map:  horizontal → sigmoid bend → line of slope m
# ---------------------------------------------------------------------
def f(x, m=1.0, k=1.0, x0=0.0, a=0.0, y0=0.0):
    """
    f(x) = y0 + m*(x - a) / (1 + exp[-k*(x - x0)])
    Parameters
    ----------
    x : float (or NumPy array)
    m : final slope on the right (>0)
    k : steepness of the bend (>0)
    x0: x‑coordinate of the bend’s midpoint
    a : point where the linear part is ‘anchored’
    y0: left‑hand horizontal level
    """
    s = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    return y0 + m * (x - a) * s


# ---------------------------------------------------------------------
# Analytic inverse via Lambert‑W
# ---------------------------------------------------------------------
def f_inv(y, m=1.0, k=1.0, x0=0.0, a=0.0, y0=0.0):
    """
    Inverts y = f(x) for x (principal branch, monotone case).
    Returns a float; for array input wrap this with NumPy’s vectorize.
    """
    B = (y - y0) / m          # normalised output
    C = x0 - a
    z = k * B * np.exp(-k * (B - C))
    w = lambertw(z)           # principal branch (real because z>0 here)
    return a + B + (w.real / k)

def guess_sigmoid_line(xx, yy, frac_plateau=0.10):
    """
    Crude first‑guess of parameters (m, k, x0, a, y0) for
        f(x) = y0 + m * (x - a) / (1 + exp[-k(x - x0)])
    
    Parameters
    ----------
    xx, yy : 1‑D array‑like
        Sample points and values.
    frac_plateau : float in (0, 0.5)
        Fraction of points taken from each end to estimate the
        left horizontal level and the right linear trend.
    
    Returns
    -------
    dict  with keys {'m', 'k', 'x0', 'a', 'y0'}
    """

    # --- sanitise / sort -------------------------------------------------
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    n = len(xx)
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

    return dict(m=m, k=k, x0=x0, a=a, y0=y0)


par = {'m': 1.5, 'k': 1.5, 'x0': -1, 'a': -5, 'y0': -0.5}
#par = {'m': 1.5, 'k': 1.5, 'x0': -1, 'a': 0, 'y0': -0.5}

x = np.linspace(-5, 5, 200)
y = f(x, **par)
#xhat = f_inv(y, **par)

x_ = np.linspace(-5, 5, 5)
y_ = f(x_, **par)
par_hat = guess_sigmoid_line(x_, y_)
ymax = f(x, **par_hat)

plt.figure()
plt.plot(x, y)
plt.plot(x, ymax, 'r')
plt.plot(x_, y_, 'k.')
#plt.plot(xhat, y, 'k--')
plt.show()