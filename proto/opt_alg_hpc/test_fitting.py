import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def f(x: np.ndarray, a, b, c, k, q) -> np.ndarray:
    #print(f'a={a:.02f}, b={b:.02f}, c={c:.02f}, k={k:.02f}, q={q:.02f}')
    y = c + a / ((1 + np.exp(-k * (x - b))) ** q)
    return y

def g(x: np.ndarray, a, b, c, k) -> np.ndarray:
    #print(f'a={a:.02f}, b={b:.02f}, c={c:.02f}, k={k:.02f}')
    y = c + a / (1 + np.exp(-k * (x - b)))
    return y

def load_data():
# =============================================================================
#     fpath_in = (r"D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc\data"
#                 r"\test_opt_hpc_batch\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0_alpha=1_2"
#                 r"\info\Ru_Rc_req_0_6.pkl")    
# =============================================================================
    fpath_in = (r"D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc\data"
                r"\test_opt_hpc_batch\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.1_alpha=1_2"
                r"\info\Ru_Rc_req_6_5.pkl")
    with open(fpath_in, 'rb') as fid:
        data = pickle.load(fid)
        
    pop_num = 3
    xx = data['Ru'][pop_num, :]
    yy = data['Rc'][pop_num, :]
    return xx, yy

def gen_data(a, b, c, k, q, xmin, xmax, nx):
    xx = np.linspace(xmin, xmax, nx)
    yy = f(xx, b, c, k, q)
    return xx, yy

def _get_bounds(bounds: Dict | None = None) -> Tuple[List[float], List[float]]:
    bounds = bounds or {}
    #par_names = ['a', 'b', 'c', 'k', 'q']
    par_names = ['a', 'b', 'c', 'k']
    bounds_ = {p: (-np.inf, np.inf) for p in par_names}
    for key, val in bounds.items():
        bounds_[key] = val
    low = [bounds_[p][0] for p in par_names]
    high = [bounds_[p][1] for p in par_names]
    return low, high


xx, yy = load_data()

dy = yy[1:] - yy[:-1]
n = np.argmax(dy)
x01, y01 = xx[n], yy[n]
x02, y02 = xx[n + 1], yy[n + 1]
x1, y1 = xx[0], yy[0]
x2, y2 = xx[-1], yy[-1]
q0 = 1
a0 = np.abs(y2 - y1)
b0 = (x01 + x02) / 2
for _ in range(5):
    k0 = 4 * (y02 - y01) / (x02 - x01) / a0
    S = lambda x: f(x, 1, b0, 0, k0, q0)
    a0 = (y2 - y1) / (S(x2) - S(x1))
    c0 = y1 - a0 * S(x1)
#k0 = 4 * (y02 - y01) / (x02 - x01) / a0
#c0 = yy.min()

xx_ = np.linspace(xx.min(), xx.max(), 200)
yy0_ = f(xx_, a0, b0, c0, k0, q0)

#sigma = np.clip(yy ** 0.5, 0.1, 5)
bounds = {
    #'q': (1, 1.01)
}

try:
    #par0 = (a0, b0, c0, k0, q0)
    par0 = (a0, b0, c0, k0)
    print('==== Optimization ====')
    par, _ = curve_fit(
        g, xx, yy, p0=par0, bounds=_get_bounds(bounds), nan_policy='omit',
        max_nfev=1000, verbose=2,
        ftol=1e-3, xtol=1e-4, method='trf'
        #sigma=sigma, absolute_sigma=True
    )
    #a, b, c, k, q = par
    a, b, c, k = par
    q = 1
except Exception as e:
    print(e)
    a, b, c, k, q = [np.nan] * 5

yy_ = f(xx_, a, b, c, k, q)

plt.figure()
plt.plot(xx_, yy0_)
plt.plot(xx_, yy_)
plt.plot(xx, yy, '.')