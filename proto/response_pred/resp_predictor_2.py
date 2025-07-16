import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm

from rank1_hess import estimate_rank1_hessian
from rate_model import RateModel, RateModelWC


class ResponsePredictor_2:

    model: RateModel
    h0: np.ndarray   # (npops x 1)

    dt: float
    nsteps: int

    r0: np.ndarray   # (npops x 1)

    J1: np.ndarray   # J1kn = dfk / drn
    Q1: np.ndarray   # Q1n = dfn / dhn

    J2: np.ndarray   # J2kmn = d2fk / (drm * drn)

    def __init__(
            self,
            model: RateModel | None = None,
            h0: np.ndarray | None = None,
            dt: float = 0.5,
            nsteps: int = 20
            ):
        self.model = model
        self.h0 = h0
        self.dt = dt
        self.nsteps = nsteps
        self._reset()
    
    def _reset(self):
        self.r0 = None
        self.J1 = None
        self.Q1 = None
        self.J2 = None
    
    def set_model(self, model: RateModel):
        self.model = model
        self.h0 = None
        self._reset()
    
    def set_h0(self, h0: np.ndarray):
        self.h0 = h0
        self._reset()

    def _calc_r0(self) -> None:
        """Find unperturbed steady-state. """
        N = self.model.npops    
        self.r0 = self.model.run(self.h0, r0=np.zeros((N, 1)),
                                 dt=self.dt, nsteps=self.nsteps)
    
    def _calc_J1(self, dr: float) -> None:
        """Calculate J1: J1kn = dfk / drn. """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        D = np.eye(N, N)
        self.J1 = np.zeros((N, N))
        for k in range(N):   # population index
            for n in range(N):   # axis to perturb
                dr_n = D[:, [n]] * dr   # perturbations along the n-th axis
                r_pert_p = self.model.run_1pop(k, self.h0[k], self.r0 + dr_n, *sim_par)
                r_pert_n = self.model.run_1pop(k, self.h0[k], self.r0 - dr_n, *sim_par)
                self.J1[k, n] = (r_pert_p - r_pert_n) / (2 * dr)

    def _calc_Q1(self, dh: float) -> None:
        """Calculate Q1: Q1n = dfn / dhn. """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        self.Q1 = np.zeros((N, 1))
        for n in range(N):   # population index
            r_pert_p = self.model.run_1pop(n, self.h0[n] + dh, self.r0, *sim_par)
            r_pert_n = self.model.run_1pop(n, self.h0[n] - dh, self.r0, *sim_par)
            self.Q1[n] = (r_pert_p - r_pert_n) / (2 * dh)
    
    def _calc_J2(self, dr: float) -> None:
        """Calculate J2: J2kmn = d2fk / (drm * drn). """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        D = np.eye(N, N)
        self.J2 = np.zeros((N, N, N))
        for k in range(N):
            for m in range(N):
                for n in range(N):
                    dr_m = D[:, [m]] * dr
                    dr_n = D[:, [n]] * dr
                    r_pert_pp = self.model.run_1pop(k, self.h0[k], self.r0 + dr_m + dr_n, *sim_par)
                    r_pert_pn = self.model.run_1pop(k, self.h0[k], self.r0 + dr_m - dr_n, *sim_par)
                    r_pert_np = self.model.run_1pop(k, self.h0[k], self.r0 - dr_m + dr_n, *sim_par)
                    r_pert_nn = self.model.run_1pop(k, self.h0[k], self.r0 - dr_m - dr_n, *sim_par)
                    self.J2[k, m, n] = (r_pert_pp - r_pert_pn - r_pert_np + r_pert_nn) / (4 * dr**2)
    
    def _calc_J2_estim(self, dr: float) -> None:
        """Calculate J2: J2kmn = d2fk / (drm * drn). """
        N = self.model.npops
        sim_par = self.dt, self.nsteps
        self.J2 = np.zeros((N, N, N))
        for k in range(N):
            F = lambda r: self.model.run_1pop(k, self.h0[k], r, *sim_par)
            self.J2[k, :, :] = estimate_rank1_hessian(F, self.r0, dr)

    def train(self, dh: float, dr: float) -> None:        
        self._calc_r0()        
        self._calc_Q1(dh)
        self._calc_J1(dr)
        self._calc_J2(dr)

    """ def predict_r(
            self,
            Dh: np.ndarray,   # (npops, 1) 
            dh_train: float | None = None,
            dr_train: float | None = None
            ) -> np.ndarray:
        
        N = self.model.npops
        Dh = Dh.reshape((N, 1))
        
        if dh_train is None:
            #dh_train = np.linalg.norm(Dh, 1)
            Q = self._get_Q_by_Dh(Dh)
        else:
            dh_train = self._clip_dh(dh_train)
            Q = self.Q.interp(dh=dh_train).values
        
        Dr_pre = Q @ Dh
        Dr_post = Dr_pre

        if dr_train is None:
            #dr_train = dh_train * np.sqrt(np.mean(np.diag(Q)**2))
            #dr_train = dh_train * np.mean(np.diag(Q))
            for n in range(10):
                J = self._get_J_by_Dr(Dr_post)
                Dr_post = np.linalg.inv(np.eye(N) - J) @ Dr_pre
                #P = self._get_P_by_Dr(Dr_post)
                #Dr_post = P @ Dr_pre
        else:
            dr_train = self._clip_dr(dr_train)
            #J = self.J.interp(dr=dr_train).values
            #Dr_post = np.linalg.inv(np.eye(N) - J) @ Dr_pre
            P = self.P.interp(dr=dr_train).values
            Dr_post = P @ Dr_pre

        r_hat = self.r0 + Dr_post
        return r_hat """



