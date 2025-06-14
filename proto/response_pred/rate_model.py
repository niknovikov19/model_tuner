from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


def _to_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x)
    else:
        return np.array(x).reshape(shape)


class RateModelBase(ABC):

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
             pop_num: int | None
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

        # Initial state
        R[:, 0] = r0[:, None]

        # To column vector
        h = h[:, None]

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[:, n - 1]
            r_hat = self.gain_func(self.W @ r + h[:, None])
            R[:, n] = r + (r_hat - r) * dt / self.tau
        
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

        # Initial state
        R[0] = r0[pop_num]

        # To column vector
        r0 = r0[:, None]

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[n - 1]
            r_hat = self.gain_func(self.W[pop_num, :] @ r0 + h)
            R[n] = r + (r_hat - r) * dt / self.tau
        
        # Convert the result to xarray
        R = xr.DataArray(
            R.reshape((1, -1)),
            dims=['pop', 'time'],
            coords={'pop': pop_num, 'time': tvec},
        )
        return R
        

class RateModelWC(RateModelBase):

    gain_slope: np.ndarray   # (npops, 1)
    gain_center: np.ndarray   # (npops, 1)
    
    def __init__(
            self,
            W: np.ndarray,
            tau: np.ndarray | float,
            gain_slope: np.ndarray | float,
            gain_center: np.ndarray | float
            ):
        super().__init__(W, tau)
        self.gain_slope = _to_array(gain_slope, (self.npops, 1))
        self.gain_center = _to_array(gain_center, (self.npops, 1))

    @abstractmethod
    def gain(self,
             mu: np.ndarray | float,
             pop_num: int | None
             ) -> np.ndarray | float:
        pass
    
    def gain_func(self, x: float | np.ndarray) -> float | np.ndarray:
        k, xc  = self.gain_slope, self.gain_center
        return self.rmax / (1 + np.exp(-k * (x - xc)))
    
    
    
    def run_1pop(self,
            pop_num: int,
            h_input: float,
            r0: np.ndarray,
            alpha: float = 1,
            nsteps: int = 10,
            ) -> tuple[float, np.ndarray]:
        r0 = r0.reshape(-1, 1)
        W = self.weights_mat[pop_num, :]
        R = np.zeros((1, nsteps))
        R[0] = r0[pop_num]
        r = r0.copy()
        for n in range(1, nsteps):
            rpop = R[n - 1]
            r[pop_num] = rpop
            rpop_new = self.gain_func(W @ r + h_input)
            R[n] = (1 - alpha) * rpop + alpha * rpop_new
        return R[-1], R