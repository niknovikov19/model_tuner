from abc import ABC, abstractmethod

import numpy as np


class RateModelBase(ABC):

    npops: int
    W: np.ndarray   # (npops, npops)
    tau: np.ndarray   # (npops, 1)

    def __init__(self, W: np.ndarray, tau: np.ndarray):
        # TODO: checks
        self.W = W
        self.tau = tau
        self.npops = W.shape[0]
    
    @abstractmethod
    def gain(self, mu: np.ndarray) -> np.ndarray:
        pass

    def simulate(self,
            h: np.ndarray,   # input: (npops, 1)
            r0: np.ndarray,   # initial state: (npops, 1)
            dt: float = 1,
            nsteps: int = 10
            ) -> np.ndarray:   # (npops, 1)

        # The reult will be stored here
        R = np.zeros((self.npops, nsteps))

        # Initial state
        R[:, 0] = r0[:, None]

        # Iterate to find the steady state
        for n in range(1, nsteps):
            r = R[:, n - 1]
            r_hat = self.gain_func(self.W @ r + h[:, None])
            R[:, n] = r + (r_hat - r) * dt / self.tau
        
        self.R = R
        return R[:, -1]

        

class RateModel:
    def __init__(
            self,
            weights_mat: np.ndarray | None = None,  
            gain_slope: float = 1,
            gain_center: float = 10,
            rmax: float = 100
            ):
        if not self.weights_mat:
            npops = 10
            weights_mat = np.random.rand((npops, npops))
        self.weights_mat = weights_mat
        self.gain_slope = gain_slope
        self.gain_center = gain_center
        self.rmax = rmax
        self.npops = self.weights_mat.shape[0]
    
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