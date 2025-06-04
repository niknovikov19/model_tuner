from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from pathlib import Path
import sys
from typing import Dict, List, Literal, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr

from model_tuner.opt.regimes import (
    PopRegime1D, NetRegime1D, NetRegime1DList
)
from model_tuner.opt.ir_mappers import NetIRMapper
from model_tuner.opt.uc_mappers import PopUCMapper1D, NetUCMapper1D

from model_tuner.main import UCMapFitParams, init_uc_mapper
from model_tuner.utils import load_yaml

from .rotated_step import rot_step


warnings.filterwarnings('ignore')


class OptStrategy(Enum):
    STEP_TO_NEW = 'step_to_new'


@dataclass
class OptStrategyParams:
    opt_strategy: OptStrategy = OptStrategy.STEP_TO_NEW
    alpha_Ru: float = 1   # (initial) step size by Ru
    alpha_Rc: float = 1      # (initial) step size by Rc
    alpha_mult_Ru: float = 0.8   # Ru step multiplier for auto-decrease
    alpha_mult_Rc: float = 0.8   # Rc step multiplier for auto-decrease
    alpha_min: float = 0.01   # min. step size for auto-decrease
    #dmax_rot: float = 0.1   # max. allowed value of 1 minus dot product
    #                        # between rotated and original step directions
    steps_by_pop: bool = False   # individual step for each pop.
    auto_decrease_step: bool = False


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

    # I-R mapper
    ir_mapper: NetIRMapper

    # Optimization strategy
    opt_strategy: OptStrategy
    opt_strategy_params: OptStrategyParams

    # Current iteration
    iter_num: int

    def __init__(
            self,
            uc_map_params: UCMapFitParams,
            pop_names: list[str],
            rr_base: float | np.ndarray | Dict[str, float],
            pfr_vec: list[float] | np.ndarray,
            n_iter: int,
            ir_mapper: NetIRMapper,
            #regime_types: tuple[type, type, type] | None = None,
            opt_strategy_params: OptStrategyParams = OptStrategyParams()
            ) -> None:
        
        self.uc_map_params = uc_map_params
        self.pop_names = list(pop_names)

        if isinstance(rr_base, dict):
            self.rr_base = np.array(
                [rr_base[pop] for pop in self.pop_names])
        else:
            self.rr_base = np.array(rr_base)

        self.pfr_vec = np.array(pfr_vec)
        self.n_iter = n_iter
        self.ir_mapper = ir_mapper
        #self._set_regime_types(regime_types)
        self.opt_strategy = opt_strategy_params.opt_strategy
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
    
    def _init_zero_iter(
            self,
            uc_mapper_0: NetUCMapper1D | Literal['auto'] | None = None,
            Ru_step_0: xr.DataArray | None = None,
            Rc_step_0: xr.DataArray | None = None
            ) -> None:

        # Check the correct pops. and pfr values order
        if Rc_step_0 is not None:
            Rc_step_0 = Rc_step_0.sel(pop=self.pop_names, pfr=self.pfr_vec)
        if Ru_step_0 is not None:
            Ru_step_0 = Ru_step_0.sel(pop=self.pop_names, pfr=self.pfr_vec)
        
        # Initialize UC mapper
        if uc_mapper_0 is None:
            # Set UC mapper to identity
            self.uc_mappers[0] = init_uc_mapper(self.uc_map_params)
        elif uc_mapper_0 == 'auto':
            if (Rc_step_0 is not None) and (Ru_step_0 is not None):
                # Fit UC mapper to the zero step data
                self.uc_mappers[0] = (
                    self._fit_net_uc_mapper_from_data(Ru_step_0, Rc_step_0))
            else:
                # Set UC mapper to identity
                self.uc_mappers[0] = init_uc_mapper(self.uc_map_params)
        else:
            # Use the provided UC mapper
            self.uc_mappers[0] = uc_mapper_0
        
        # Set the 0-th Rc step
        if Rc_step_0 is not None:
            self.step_data['Rc'].loc[{'iter': 0}] = Rc_step_0
        else:
            self.step_data['Rc'].loc[{'iter': 0}] = deepcopy(self.Rc0)
        
        # Set the 0-th Ru step
        if Ru_step_0 is not None:
            self.step_data['Ru'].loc[{'iter': 0}] = Ru_step_0
        else:
            Rc0_lst = NetRegime1DList.from_xr(
                self.step_data['Rc'].sel(iter=0))
            Ru0_lst = self.uc_mappers[0].Rc_to_Ru(Rc0_lst)
            #Ru0_lst.__class__ = NetRegime1DList
            self.step_data['Ru'].loc[{'iter': 0}] = (
                Ru0_lst.to_xr('pfr', self.pfr_vec))
    
    def begin(
            self,
            uc_mapper_0: NetUCMapper1D | Literal['auto'] | None = None,
            Ru_step_0: xr.DataArray | None = None,
            Rc_step_0: xr.DataArray | None = None
            ) -> None:
        """Prepare for the 1-st iteration. """
        self.Rc0 = self._calc_Rc0()

        self.sim_data = self._alloc_sim_data()
        self.step_data = self._alloc_step_data()
        
        self.uc_mappers = [None] * self.n_iter

        self._init_zero_iter(uc_mapper_0, Ru_step_0, Rc_step_0)

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
    
    def _fit_net_uc_mapper_from_data(
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
    
    def _fit_pop_uc_mapper_from_data(
            self,
            Ru: xr.DataArray,   # (1 x pfr)
            Rc: xr.DataArray,   # (1 x pfr)
            ) -> PopUCMapper1D:
        # Initialize pop UC mapper
        uc_mapper = PopUCMapper1D(
            map_type=self.uc_map_params.map_type,
            map_params=self.uc_map_params.map_params
        )
        if self.uc_map_params.use_fit_weights:
            raise NotImplementedError(
                'Fitting with weights is not implemented for PopUCMapper1D'
            )
        # Fit pop UC mapper to (Ru, Rc)
        uc_mapper.fit_from_data(
            values_in=Ru.values,
            values_out=Rc.values,
            fit_params=self.uc_map_params.map_fit_params,
            bounds=self.uc_map_params.fit_param_bounds,
            weights=None
        )
        return uc_mapper
    
    def _is_pop_uc_mapping_valid(
            self,
            uc_mapper: PopUCMapper1D,
            pop_name: str
            ) -> bool:        
        # Check whether uc_mapper itself is valid
        # (identity or successfully fiited)
        if not uc_mapper.is_valid():
            logging.warning(f'UC mapper for {pop_name} is invalid (fitting failed)')
            return False
        
        # Check whether UC mapping can invert Rc0
        Ru = uc_mapper.Rc_to_Ru(self.Rc0.sel(pop=pop_name))
        if not all(Ru_.is_valid() for Ru_ in Ru):
            logging.warning(f'UC mapper for {pop_name} cannot convert Rc0 to Ru')
            return False
        
        # Check whether IR mapping can invert Ru
        Iu = [self.ir_mapper[pop_name].R_to_I(Ru_) for Ru_ in Ru]
        if not all(Iu_.is_valid() for Iu_ in Iu):
            logging.warning(f'IR mapper for {pop_name} cannot convert Ru to Iu')
            return False

        return True        
    
    def _fit_uc_step_to_new(self) -> tuple[NetUCMapper1D,
                                           xr.DataArray,
                                           xr.DataArray]:
        # Relative size of the new step (from prev step to sim result)           
        alpha_Rc_0 = self.opt_strategy_params.alpha_Rc
        alpha_Ru_0 = self.opt_strategy_params.alpha_Ru
        alpha_mult_Rc = self.opt_strategy_params.alpha_mult_Rc
        alpha_mult_Ru = self.opt_strategy_params.alpha_mult_Ru
        if self.opt_strategy_params.auto_decrease_step:
            alpha_min = self.opt_strategy_params.alpha_min
        else:
            alpha_min = np.inf

        # Initialize the new step by the previous one.
        # It will be updated for the pops. with successful UC fitting
        Ru_new_all = self._get_prev_Ru_step().copy()
        Rc_new_all = self._get_prev_Rc_step().copy()

        # Initialize the new UC mapper by the previous one.
        # It will be updated for the pops. with successful UC fitting
        uc_mapper = deepcopy(self.uc_mappers[self.iter_num - 1])
        uc_fit_ok = False

        for pop in self.pop_names:
            # Previous step
            Ru_prev = self._get_prev_Ru_step().sel(pop=pop)
            Rc_prev = self._get_prev_Rc_step().sel(pop=pop)

            # Recent simulation result
            Ru_sim = self._get_cur_Ru_sim().sel(pop=pop)
            Rc_sim = self._get_cur_Rc_sim().sel(pop=pop)

            # One iteration or a loop with decreasing alpha
            alpha_Ru = alpha_Ru_0
            alpha_Rc = alpha_Rc_0
            while True:

                # New step
                Ru_new = alpha_Ru * Ru_sim + (1 - alpha_Ru) * Ru_prev
                Rc_new = alpha_Rc * Rc_sim + (1 - alpha_Rc) * Rc_prev

                # Fit UC mapper for the pop.
                pop_uc_mapper = self._fit_pop_uc_mapper_from_data(Ru_new, Rc_new)
                
                # Check whether the new UC mapping is valid
                # (fitting succeeded, and Rc0->Ru->Iu conversion is possible)
                if self._is_pop_uc_mapping_valid(pop_uc_mapper, pop):
                    # Store the new step and the UC mapper fitted to it
                    uc_mapper[pop] = pop_uc_mapper
                    Ru_new_all.loc[{'pop': pop}] = Ru_new
                    Rc_new_all.loc[{'pop': pop}] = Rc_new
                    uc_fit_ok = True
                    break
                
                # Cannot decrease alpha anymore
                if (alpha_Ru < alpha_min) or (alpha_Rc < alpha_min):
                    if not self.opt_strategy_params.steps_by_pop:
                        raise RuntimeError(
                            f'UC mapping for {pop} failed: '
                            'Rc0->Ru->Iu conversion impossible')
                    break   # this pop will remain at the previous step

                # Decrease alpha
                alpha_Ru *= alpha_mult_Ru
                alpha_Rc *= alpha_mult_Rc
                logging.warning(
                    f'Decrease alpha: ({alpha_Ru:.04f}, {alpha_Rc:.04f})'
                )
        
        if not uc_fit_ok:
            raise RuntimeError(
                'UC mapping failed for all pops (fitting failed or Rc0->Ru->Iu conversion impossible')

        return uc_mapper, Ru_new_all, Rc_new_all
    
    def _fit_uc_step_to_new_rot(self) -> tuple[NetUCMapper1D,
                                               xr.DataArray,
                                               xr.DataArray]:
        # Recent simulation result
        Ru_sim = self._get_cur_Ru_sim()
        Rc_sim = self._get_cur_Rc_sim()

        # Previous step
        Ru_prev = self._get_prev_Ru_step()
        Rc_prev = self._get_prev_Rc_step()

        # Ru step
        alpha_Ru = self.opt_strategy_params.alpha_Ru
        Ru_new = alpha_Ru * Ru_sim + (1 - alpha_Ru) * Ru_prev

        # Non-rotated Rc step vector
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

        return uc_mapper, Ru_new, Rc_new_rot
    
    def fit_uc_mapper(self) -> None:
        """Fit UC mapper for the current iteration. """

        # Choose next step Rc' and fit UC mapper for (Ru_sim, Rc')
        if self.opt_strategy == OptStrategy.STEP_TO_NEW:
            res = self._fit_uc_step_to_new()
        elif self.opt_strategy == OptStrategy.STEP_TO_NEW_ROT:
            res = self._fit_uc_step_to_new_rot()
        else:
            raise ValueError(f'Unknown optimization strategy: {self.opt_strategy}')
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
    
    
#logging.basicConfig(level=logging.WARNING, force=True)