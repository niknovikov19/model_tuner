from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


def _to_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x)
    else:
        return np.array(x).reshape(shape)


class RateModel(ABC):

    npops: int
    W: np.ndarray   # (npops, npops)
    tau: np.ndarray   # (npops, 1)

    def __init__(
            self,
            W: np.ndarray,   # (npops, npops)
            tau: np.ndarray | float   # (npops, 1)
            ):
        # TODO: checks
        self.W = W
        self.npops = W.shape[0]
        self.tau = _to_array(tau, (self.npops, 1))
    
    @abstractmethod
    def gain(self,
             mu: np.ndarray | float,
             pop_num: int | None = None
             ) -> np.ndarray | float:
        pass

    def run(self,
            h: np.ndarray,   # input: (npops, 1)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> xr.DataArray:   # (npops, nsteps)

        # The result will be stored here
        R = np.zeros((self.npops, nsteps))

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        # To column vectors
        r0 = r0.reshape((self.npops, 1))
        h = h.reshape((self.npops, 1))

        # Initial state
        R[:, [0]] = r0

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[:, [n - 1]]
            r_hat = self.gain(self.W @ r + h)
            R[:, [n]] = r + (r_hat - r) * dt / self.tau
        
        # Convert the result to xarray
        R = xr.DataArray(
            R,
            dims=['pop', 'time'],
            coords={'pop': np.arange(self.npops), 'time': tvec},
        )
        return R
    
    def run_1pop(self,
            pop_num: int,
            h: float,
            r0: np.ndarray,   # surrogate rates (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> xr.DataArray:   # (1, nsteps)
        
        # The result will be stored here
        R = np.zeros(nsteps)

        # Time bins
        tvec = np.arange(0, nsteps * dt, dt)

        # To column vector
        r0 = r0.reshape((self.npops, 1))

        # Initial state
        R[0] = r0[pop_num]

        # Steady-state
        r_hat = self.gain(
            self.W[pop_num, :] @ r0 + h, pop_num)

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[n - 1]
            R[n] = r + (r_hat - r) * dt / self.tau[pop_num]
        
        # Convert the result to xarray
        R = xr.DataArray(
            R.reshape((1, -1)),
            dims=['pop', 'time'],
            coords={'pop': [pop_num], 'time': tvec},
        )
        return R
        

class RateModelWC(RateModel):

    rmax: np.ndarray   # (npops, 1)
    gain_slope: np.ndarray   # (npops, 1)
    gain_center: np.ndarray   # (npops, 1)
    
    def __init__(
            self,
            W: np.ndarray,
            tau: np.ndarray | float,
            rmax: np.ndarray | float,
            gain_slope: np.ndarray | float,
            gain_center: np.ndarray | float
            ):
        super().__init__(W, tau)
        self.rmax = _to_array(rmax, (self.npops, 1))
        self.gain_slope = _to_array(gain_slope, (self.npops, 1))
        self.gain_center = _to_array(gain_center, (self.npops, 1))

    def gain(self,
             mu: np.ndarray | float,   # (npops x nsamples) or (1 x nsamples)
             pop_num: int | None = None
             ) -> np.ndarray | float:
        rmax, k, c  = self.rmax, self.gain_slope, self.gain_center
        if pop_num is None:
            mu = mu.reshape((self.npops, -1))
            return rmax / (1 + np.exp(-k * (mu - c)))
        else:
            mu = mu.reshape((1, -1))
            rmax, k, c = rmax[pop_num], k[pop_num], c[pop_num]
            return rmax / (1 + np.exp(-k * (mu - c)))    
