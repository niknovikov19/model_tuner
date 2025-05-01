from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr

from model_tuner.opt.regimes import (
    PopRegime1D, NetRegime1D, NetRegime1DList
)
from model_tuner.opt.ir_mappers import NetIRMapperWC
from model_tuner.opt.uc_mappers import NetUCMapper1D
from model_tuner.opt.wc import ModelDescWC
from model_tuner.opt.wc import wc_gain, run_wc_model

from model_tuner.main import UCMapFitParams, init_uc_mapper
from model_tuner.utils import load_yaml

from rotated_step import rot_step


warnings.filterwarnings('ignore')


class OptStrategy(Enum):
    STEP_TO_NEW = auto()
    STEP_TO_NEW_AUTOSZ = auto()
    STEP_TO_NEW_ROT = auto()


@dataclass
class OptStrategyParams:
    alpha: float = 1      # (initial) step size by Rc
    alpha_Ru: float = 1   # (initial) step size by Ru
    alpha_mult: float = 0.8   # step size multiplier for STEP_TO_NEW_AUTOSZ
    alpha_min: float = 0.01   # min. step size for STEP_TO_NEW_AUTOSZ
    dmax_rot: float = 0.1   # max. allowed value of 1 minus dot product
                            # between rotated and original step directions


class UCOptimizer:

    # Parameters
    uc_map_params: UCMapFitParams
    pop_names: list[str]
    rr_base: np.ndarray   # (pop x 1)
    pfr_vec: np.ndarray   # (pfr x 1)
    n_iter: int

    # Regime types to work with:
    # (PopRegime1D, NetRegime1D, NetRegime1DList) or subclasses
    #regime_types: Dict[str, type]

    # Target regimes
    Rc0: xr.DataArray   # (pop x pfr)

    # Iterations: (pop x pfr x iter)
    sim_data: xr.Dataset   # Simulated (Ru, Rc) points
    step_data: xr.Dataset  # Rc points chosen as opt. steps

    # UC mappers resulting from each step
    uc_mappers: list[NetUCMapper1D]

    # Optimization strategy
    opt_strategy: OptStrategy
    opt_strategy_params: OptStrategyParams

    # Current iteration
    iter_num: int

    def __init__(
            self,
            uc_map_params: UCMapFitParams,
            pop_names: list[str],
            rr_base: float | np.ndarray,
            pfr_vec: list[float] | np.ndarray,
            n_iter: int,
            #regime_types: tuple[type, type, type] | None = None,
            opt_strategy: OptStrategy = OptStrategy.STEP_TO_NEW,
            opt_strategy_params: OptStrategyParams = OptStrategyParams()
            ) -> None:
        self.uc_map_params = uc_map_params
        self.pop_names = pop_names
        self.rr_base = np.array(rr_base)
        self.pfr_vec = np.array(pfr_vec)
        self.n_iter = n_iter
        #self._set_regime_types(regime_types)
        self.opt_strategy = opt_strategy
        self.opt_strategy_params = opt_strategy_params

        self.begin()
    
    """ def _set_regime_types(
            self,
            regime_types: tuple[type, type, type] | None
            ) -> None:
        regime_types_base = (PopRegime1D, NetRegime1D, NetRegime1DList)
        if regime_types is None:
            regime_types = regime_types_base
        for t, t_base in zip(regime_types, regime_types_base):
            if not issubclass(t, t_base):
                raise TypeError(
                    f"Regime type {t} is not a subclass of {t_base}"
                )
        regime_keys = ('pop', 'net', 'net_list')
        for key, t in zip(regime_keys, regime_types):
            self.regime_types[key] = t   """          

    def _get_shape(self) -> tuple[float, float, float]:
        return len(self.pop_names), len(self.pfr_vec), self.n_iter
    
    def _alloc_sim_data(self) -> xr.Dataset:
        # Allocate storage for sim (Ru, Rc) points
        Ru = xr.DataArray(
            np.full(self._get_shape(), np.nan),
            dims=['pop', 'pfr', 'iter'],
            coords={'pop': self.pop_names,
                    'pfr': self.pfr_vec,
                    'iter': np.arange(self.n_iter)}
        )
        Rc = Ru.copy()
        return xr.Dataset({'Ru': Ru, 'Rc': Rc})
    
    def _alloc_step_data(self) -> xr.Dataset:
        # Allocate storage for step Rc values
        Ru = xr.DataArray(
            np.full(self._get_shape(), np.nan),
            dims=['pop', 'pfr', 'iter'],
            coords={'pop': self.pop_names,
                    'pfr': self.pfr_vec,
                    'iter': np.arange(self.n_iter)}
        )
        Rc = Ru.copy()
        return xr.Dataset({'Ru': Ru, 'Rc': Rc})

    def _calc_Rc0(self) -> xr.DataArray:
        # Calculate Rc0 as scaled versions of rr_base by pfr_vec
        return xr.DataArray(
            self.rr_base[:, None] * self.pfr_vec[None, :],
            dims=['pop', 'pfr'],
            coords={'pop': self.pop_names, 'pfr': self.pfr_vec}
        )
    
    def _init_zero_iter(self) -> None:
        """Set 0-th step to Rc0 and 0-th uc_mapper to identity. """
        self.step_data['Ru'].loc[{'iter': 0}] = self.Rc0
        self.step_data['Rc'].loc[{'iter': 0}] = self.Rc0
        self.uc_mappers[0] = init_uc_mapper(self.uc_map_params)
    
    def begin(self) -> None:
        """Prepare for the 1-st iteration. """
        self.Rc0 = self._calc_Rc0()

        self.sim_data = self._alloc_sim_data()
        self.step_data = self._alloc_step_data()
        
        self.uc_mappers = [None] * self.n_iter

        self._init_zero_iter()

        self.iter_num = 1
    
    def suggest_Ru(self) -> NetRegime1DList:
        """Choose Ru to put into a simulation for the current iteration. """
        Rc0_lst = NetRegime1DList.from_xr(self.Rc0)
        uc_mapper = self.uc_mappers[self.iter_num - 1]
        return uc_mapper.Rc_to_Ru(Rc0_lst)
    
    def store_sim_result(
            self,
            Ru_lst: NetRegime1DList,
            Rc_lst: NetRegime1DList
            ) -> None:
        """Store simulation result for the current iteration. """
        self.sim_data['Ru'].loc[{'iter': self.iter_num}] = (
            Ru_lst.to_xr('pfr', self.pfr_vec)
        )
        self.sim_data['Rc'].loc[{'iter': self.iter_num}] = (
            Rc_lst.to_xr('pfr', self.pfr_vec)
        )
    
    def _get_prev_Ru_step(self) -> xr.DataArray:
        return self.step_data['Ru'].isel(
            iter=(self.iter_num - 1), drop=True)
    def _get_cur_Ru_step(self) -> xr.DataArray:
        return self.step_data['Ru'].isel(
            iter=self.iter_num, drop=True)
    def _get_prev_Rc_step(self) -> xr.DataArray:
        return self.step_data['Rc'].isel(
            iter=(self.iter_num - 1), drop=True)
    def _get_cur_Rc_step(self) -> xr.DataArray:
        return self.step_data['Rc'].isel(
            iter=self.iter_num, drop=True)
    
    def _get_cur_Ru_sim(self) -> xr.DataArray:
        return self.sim_data['Ru'].isel(
            iter=self.iter_num, drop=True)
    def _get_cur_Rc_sim(self) -> xr.DataArray:
        return self.sim_data['Rc'].isel(
            iter=self.iter_num, drop=True)
    
    def _fit_uc_mapper_from_data(
            self,
            Ru: xr.DataArray,   # (pop x pfr)
            Rc: xr.DataArray,   # (pop x pfr)
            verbose: bool = False
            ) -> NetUCMapper1D:
        uc_mapper = init_uc_mapper(self.uc_map_params)
        Ru_lst = NetRegime1DList.from_xr(Ru)
        Rc_lst = NetRegime1DList.from_xr(Rc)
        uc_mapper.fit_from_data(
            Ru_lst, Rc_lst,
            fit_params=self.uc_map_params.map_fit_params,
            bounds=self.uc_map_params.fit_param_bounds,
            verbose=verbose
        )
        return uc_mapper
    
    def _fit_uc_step_to_new(self) -> tuple[NetUCMapper1D,
                                           xr.DataArray,
                                           xr.DataArray]:
        # Get recent simulation result
        Ru_sim = self._get_cur_Ru_sim()
        Rc_sim = self._get_cur_Rc_sim()

        # Previous step
        Ru_prev = self._get_prev_Ru_step()
        Rc_prev = self._get_prev_Rc_step()

        # Mix previous step with the recent sim result            
        alpha = self.opt_strategy_params.alpha
        alpha_Ru = self.opt_strategy_params.alpha_Ru
        Ru_new = alpha_Ru * Ru_sim + (1 - alpha_Ru) * Ru_prev
        Rc_new = alpha * Rc_sim + (1 - alpha) * Rc_prev

        # Fit UC mapper to (Ru_sim, Rc_new)
        uc_mapper = self._fit_uc_mapper_from_data(Ru_new, Rc_new)

        return uc_mapper, Ru_new, Rc_new
    
    def _fit_uc_step_to_new_autosz(self) -> tuple[NetUCMapper1D,
                                                  xr.DataArray,
                                                  xr.DataArray]:
        # Recent simulation result
        Ru_sim = self._get_cur_Ru_sim()
        Rc_sim = self._get_cur_Rc_sim()

        # Previous step
        Ru_prev = self._get_prev_Ru_step()
        Rc_prev = self._get_prev_Rc_step()

        # Initial step size
        alpha = self.opt_strategy_params.alpha
        alpha_min = self.opt_strategy_params.alpha_min

        verbose = 1

        uc_fit_ok = False
        while alpha > alpha_min:
            # Mix previous step Rc_prev with simulation result Rc_sim
            Rc_new = alpha * Rc_sim + (1 - alpha) * Rc_prev

            # Fit UC mapper to (Ru_sim, Rc_new)
            uc_mapper = self._fit_uc_mapper_from_data(Ru_sim, Rc_new)

            # Check whether the fitted mapping can invert Rc0
            Ru_lst_hat = uc_mapper.Rc_to_Ru(
                NetRegime1DList.from_xr(self.Rc0)
            )
            if all(Ru.is_valid() for Ru in Ru_lst_hat):
                if verbose:
                    print(f'Inverse mapping ok with alpha={alpha:.04f}')
                uc_fit_ok = True
                break
            else:
                if verbose:
                    print(f'Inverse mapping failed with alpha={alpha:.04f}')
                alpha *= self.opt_strategy_params.alpha_mult   # decrease alpha   

        if not uc_fit_ok:
            raise RuntimeError('UC mapping failed')
                
        return uc_mapper, Ru_sim, Rc_new
    
    def _fit_uc_step_to_new_rot(self) -> tuple[NetUCMapper1D,
                                               xr.DataArray,
                                               xr.DataArray]:
        # Recent simulation result
        Ru_sim = self._get_cur_Ru_sim()
        Rc_sim = self._get_cur_Rc_sim()

        # Previous Rc step
        Rc_prev = self._get_prev_Rc_step()

        # Non-rotated step vector
        alpha = self.opt_strategy_params.alpha
        Rc_step_0 = alpha * (Rc_sim - Rc_prev)

        # Rotate Rc_new around Rc_prev (for each pfr)
        Rc_new_rot = xr.full_like(Rc_prev, np.nan)
        Rc0_ = self.Rc0.isel(pfr=-1).values
        for pfr in self.pfr_vec:
            Rc_prev_ = Rc_prev.sel(pfr=pfr).values
            Rc_step_0_ = Rc_step_0.sel(pfr=pfr).values
            Rc_new_rot.loc[{'pfr': pfr}] = rot_step(
                a=Rc_prev_,
                c=Rc_step_0_,
                s=Rc0_,
                dmax=(1.0 - self.opt_strategy_params.dmax_rot)
            )

        # Fit UC mapper to (Ru_sim, Rc_new)
        uc_mapper = self._fit_uc_mapper_from_data(Ru_sim, Rc_new_rot)

        return uc_mapper, Ru_sim, Rc_new_rot
    
    """ def _fit_uc_step_to_new_rot_rand(self) -> tuple[NetUCMapper1D, xr.DataArray]:
        # Get recent simulation result
        Ru_sim = self._get_cur_Ru_sim()
        Rc_sim = self._get_cur_Rc_sim()
        
        # Previous step endpoint
        Rc_prev = self._get_prev_Rc_step()

        v = alpha * ()


        # Step from Rc_prev towards Rc_sim           
        alpha = self.opt_strategy_params.alpha
        Rc_new = alpha * Rc_sim + (1 - alpha) * Rc_prev

        # Fit UC mapper to (Ru_sim, Rc_new)
        uc_mapper = self._fit_uc_mapper_from_data(Ru_sim, Rc_new) """
    
    def fit_uc_mapper(self) -> NetUCMapper1D:
        """Fit UC mapper for the current iteration. """

        # Choose next step Rc' and fit UC mapper for (Ru_sim, Rc')
        if self.opt_strategy == OptStrategy.STEP_TO_NEW:
            res = self._fit_uc_step_to_new()
        elif self.opt_strategy == OptStrategy.STEP_TO_NEW_AUTOSZ:
            res = self._fit_uc_step_to_new_autosz()
        elif self.opt_strategy == OptStrategy.STEP_TO_NEW_ROT:
            res = self._fit_uc_step_to_new_rot()
        uc_mapper, Ru_step, Rc_step = res

        # Store the chosen step Rc' and fitted UC mapper for this iteration
        self.uc_mappers[self.iter_num] = uc_mapper
        self.step_data['Ru'].loc[{'iter': self.iter_num}] = Ru_step
        self.step_data['Rc'].loc[{'iter': self.iter_num}] = Rc_step

        return uc_mapper
    
    def next_iter(self) -> None:
        """Finalize current iteration and go to the next one. """
        if self.uc_mappers[self.iter_num] is None:
            self.fit_uc_mapper()
        self.iter_num += 1
    
    def finished(self) -> bool:
        return self.iter_num >= self.n_iter
    
    
