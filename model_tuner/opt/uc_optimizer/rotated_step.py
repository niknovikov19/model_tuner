from matplotlib import pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm


def line_normal(
        x: np.ndarray,   # (npoints x ndim)
        s: np.ndarray    # (1 x ndim)
        ) -> float:   # (npoints x ndim)
    """Normal from point x to a line from origin parallel to s. """
    ndim = len(s)
    s = s / norm(s)
    if x.ndim == 1:
        x = x.reshape(1, ndim)
    a = s * dot(x, s).reshape(-1, 1)
    return a - x

def line_dist(
        x: np.ndarray,   # (npoints x ndim)
        s: np.ndarray    # (1 x ndim)
        ) -> float:   # (npoints x 1)
    """Distance from point x to a line from origin parallel to s. """
    return norm(line_normal(x, s), axis=-1)

def norm_dot(x, y, axis=-1):
    nx = norm(x, axis=axis, keepdims=True)
    ny = norm(y, axis=axis, keepdims=True)
    return dot(x / nx, y / ny)

def gen_random_vectors(
        n: int,   # number of vectors to generate
        ndim: int,   # dimensionality
        r: float,   # length of the vectors
        c: np.ndarray,   # central direction (1 x ndim)
        d: float   # min. dot product between normalized x and c
        ) -> np.ndarray:   # (n x ndim)
    """Return n vectors x such that |x|=r and dot(x/|x|, c/|c|) >= d. """
    X = np.zeros((n, ndim))
    ngen = 0   # num. of generated vectors
    k = 3   # chunk size multiplier
    c /= norm(c.reshape((1, ndim)))

    while ngen < n:
        V = np.random.randn(k * n, ndim)
        V /= norm(V, axis=1, keepdims=True)
        mask = np.sum(V * c, axis=1) >= d
        V = V[mask, :]
        m = min(n - ngen, V.shape[0])
        X[ngen : ngen + m, :] = V[:m, :]
        ngen += m
    
    return X * r

def rot_step(
        a: np.ndarray,   # origin of the step
        c: np.ndarray,   # central direction of the step
        s: np.ndarray,   # line to minimize the distance to (starting from zero)
        dmax: float      # closeness of the step direction to c (<x,c> >= 1-d)
        ) -> np.ndarray:    # a + step
    """Find a+x closest to the line s, with x diretion close to c. """
    
    if np.any(np.isnan(a)):
        raise ValueError('a contains NaN values')
    if np.any(np.isnan(c)):
        raise ValueError('c contains NaN values')

    N = 100
    ndim = len(a)
    r = norm(c)
    X = gen_random_vectors(N, ndim=ndim, r=r, c=c, d=(1 - dmax))

    rr = norm(X, axis=1)
    dd = 1 - norm_dot(X, c)

    #print('min(||x|-r|) = %f' % np.min(np.abs(rr - r)))
    #print('max(1 - <x, c>) = %f' % np.max(dd))
    #print()

    hh = line_dist(a + X, s)
    n0 = np.argmin(hh)
    #h0 = hh[n0]
    x0 = X[n0, :]
    #d0 = dd[n0]

    return a + x0