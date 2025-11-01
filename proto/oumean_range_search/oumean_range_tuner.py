from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.data_proc import (
    DataKeeper,
    NetSpikesParams,
    NetRatesParams
)
from model_tuner.utils import set_qt_backend
from model_tuner.sim_manager import SimResultLocator

from bracket import JointIntervalFinder
from get_sim_rates_ import get_sim_rates


@dataclass
class OUMeanRangeParams:
    pop_names: list[str]
    ou_std_vals: list[float]
    num_ou_mean_vals: int
    ou_mean_range_start: (tuple[float, float] | 
                          dict[str, tuple[float, float]])
    rate_limits: (tuple[float, float] | 
                  dict[str, tuple[float, float]])
    tolerance: float = 0.05
    tcalc_win: tuple[float, float | None] = (1, None)


class OUMeanRangeTuner:

    # Params
    _pop_names: list[str]
    _ou_std_vals: list[float]
    _num_ou_mean_vals: int
    _ou_mean_range_start: (tuple[float, float] |
                           dict[str, tuple[float, float]])
    _rate_limits: (tuple[float, float] | 
                   dict[str, tuple[float, float]])
    _tolerance: float
    _tcalc_win: tuple[float, float]   # time window for rate calculation
    _dummy_mode: bool
    _duration: float
    
    # State
    _ou_mean_vals: dict[str, np.ndarray]   # (pop: ou_mean x 1)
    _rates: dict[str, xr.DataArray]   # (pop: ou_mean x ou_std)

    # Low-level interval finder objects
    _finders: dict[str, JointIntervalFinder]

    def __init__(self, par: OUMeanRangeParams,
                 dummy_mode=False, dummy_func=None):
        self._pop_names = par.pop_names
        self._ou_std_vals = par.ou_std_vals
        self._num_ou_mean_vals = par.num_ou_mean_vals
        self._tolerance = par.tolerance
        self._tcalc_win = par.tcalc_win
        self._dummy_mode = dummy_mode
        self._dummy_func = dummy_func or self._dummy_func_default
        #self._duration = duration

        # Set ou_mean_range_start for every pop
        if isinstance(par.ou_mean_range_start, dict):
            self._ou_mean_range_start = deepcopy(par.ou_mean_range_start)
        else:
            self._ou_mean_range_start = {pop: deepcopy(par.ou_mean_range_start)
                                         for pop in self._pop_names}
        # Set rate_limits for every pop
        if isinstance(par.rate_limits, dict):
            self._rate_limits = deepcopy(par.rate_limits)
        else:
            self._rate_limits = {pop: deepcopy(par.rate_limits)
                                 for pop in self._pop_names}
           
        # Initialize the state
        self._ou_mean_ranges = {}
        self._rates = {}

        # Initialize low-level interval finders
        self._init_finders()

        # Initial selection of ou_mean vals
        self._ou_mean_vals = self.suggest_next_ranges()
    
    def _init_finders(self):
        self._finders = {}
        for pop in self._pop_names:
            self._finders[pop] = JointIntervalFinder(
                y1=self._ou_std_vals[0],
                y2=self._ou_std_vals[-1],
                r1=self._rate_limits[pop][0],
                r2=self._rate_limits[pop][1],
                x1_start=self._ou_mean_range_start[pop][0],
                x2_start=self._ou_mean_range_start[pop][1],
                N=self._num_ou_mean_vals,
                tol_frac=self._tolerance,
                max_iters=1000,
                x2_mode="y1_ge",
            )
    
    def suggest_next_ranges(self) -> dict[str, np.ndarray]:
        res = {}
        for pop in self._pop_names:
            res[pop] = self._finders[pop].suggest_x_values()
        return res
    
    def step(self) -> None:
        self._ou_mean_vals = self.suggest_next_ranges()
    
    def final_step(self) -> None:
        for pop in self._pop_names:
            range = self._finders[pop].get_estimates()
            if (range[0] == None) or (range[1] == None):
                raise ValueError(f'Cannot perform final step: '
                                 f'range not found for pop {pop}')
            self._ou_mean_vals[pop] = np.linspace(
                range[0], range[1], self._num_ou_mean_vals)
    
    def get_ou_mean_limits(self) -> dict[str, tuple[float, float]]:
        res = {}
        for pop in self._pop_names:
            x = self._ou_mean_vals[pop]
            res[pop] = (x.min(), x.max())
        return res
    
    def is_done(self) -> bool:
        b = [finder.is_done() for finder in self._finders.values()]
        return all(b)

    def gen_sim_label(self, iter_num: int,
                      ou_std_num: int, ou_mean_num: int) -> str:
        return f'req_it_{iter_num}_std_{ou_std_num}_mean_{ou_mean_num}'
    
    def create_sim_requests(self, iter_num: int) -> dict[str, dict]:
        requests = {}
        if self._dummy_mode:
            return requests
        for ou_std_num, ou_std in enumerate(self._ou_std_vals):
            for ou_mean_num in range(self._num_ou_mean_vals):
                sim_label = self.gen_sim_label(iter_num, ou_std_num, ou_mean_num)
                req = {
                    'input': {},
                    'subnet_params': {'pops_active': self._pop_names},
                    #'duration': self._duration
                }
                for pop in self._pop_names:
                    req['input'][pop] = {
                        'ou_std': ou_std,
                        'ou_mean': self._ou_mean_vals[pop][ou_mean_num]
                    }
                requests[sim_label] = req
        return requests
    
    @staticmethod
    def _dummy_func_default(x, y, pop_n):
        rmax = 150
        xc = 0
        xmax = 0.003
        k = 1500
        r = rmax / (1 + np.exp(-(x - xc + pop_n * 0.001) * (k / (1 + y * 100))))
        r *= (x < xmax)
        return r

    def _process_sim_results_dummy(self):
        R = {}
        # Generate surrogate rates
        for n, pop in enumerate(self._pop_names):
            R[pop] = xr.DataArray(
                np.zeros((self._num_ou_mean_vals, len(self._ou_std_vals))),
                dims=['ou_mean', 'ou_std'],
                coords={'ou_mean': self._ou_mean_vals[pop],
                        'ou_std': self._ou_std_vals}
            )
            for ou_std in self._ou_std_vals:
                x = self._ou_mean_vals[pop]
                r = self._dummy_func(x, ou_std, n)
                R[pop].loc[{'ou_std': ou_std}] = r
        # Update low-level interval finders
        for pop in self._pop_names:
            self._finders[pop].process_probe_result(
                x_vals=R[pop].coords['ou_mean'].values,
                f1_vals=R[pop].isel(ou_std=0).values,
                f2_vals=R[pop].isel(ou_std=-1).values
            )
        # Update the state
        self._rates = R
        #self._ou_mean_vals = self.suggest_next_ranges()

    def process_sim_results(
            self,
            dk: DataKeeper,
            sim_res_locator: SimResultLocator,
            iter_num: int
            ) -> None:
        # Generate surrogate rates for testing
        if self._dummy_mode:
            self._process_sim_results_dummy()
            return
        # Allocate rate storage
        R = {}
        for pop in self._pop_names:
            R[pop] = xr.DataArray(
                np.zeros((self._num_ou_mean_vals, len(self._ou_std_vals))),
                dims=['ou_mean', 'ou_std'],
                coords={'ou_mean': self._ou_mean_vals[pop],
                        'ou_std': self._ou_std_vals}
            )        
        # Retrieve the rates from the simulation results
        for ou_std_num, _ in enumerate(self._ou_std_vals):
            for ou_mean_num in range(self._num_ou_mean_vals):
                print('.', end='', flush=True)
                sim_label = self.gen_sim_label(iter_num, ou_std_num, ou_mean_num)
                R_ = get_sim_rates(
                    dk, sim_res_locator,
                    sim_label,
                    spikes_calc_params=NetSpikesParams(),
                    rates_calc_params=NetRatesParams(time_limits=self._tcalc_win)
                )
                for pop in self._pop_names:
                    R[pop][dict(ou_mean=ou_mean_num,
                                ou_std=ou_std_num)] = R_[pop]
        # Update low-level interval finders
        for pop in self._pop_names:
            self._finders[pop].process_probe_result(
                x_vals=R[pop].coords['ou_mean'].values,
                f1_vals=R[pop].isel(ou_std=0).values,
                f2_vals=R[pop].isel(ou_std=-1).values
            )
        # Update the state
        self._rates = R
        #self._ou_mean_vals = self.suggest_next_ranges()

    def plot_rates(self, dirpath_figs: Path | str,
                   iter_num: int) -> None:    
        set_qt_backend()
        plt.ion()        
        for pop_name in self._pop_names:
            # Create pop subfolder
            dirpath_figs_pop = dirpath_figs / pop_name
            os.makedirs(dirpath_figs_pop, exist_ok=True)
            # Plot
            plt.figure(111)
            plt.clf()
            ou_mean_vals = self._rates[pop_name].coords['ou_mean'].values
            for ou_std in self._ou_std_vals:
                r_vals = self._rates[pop_name].sel(ou_std=ou_std).values
                plt.plot(ou_mean_vals.ravel(), r_vals.ravel(), '.-',
                         label=f'ou_std={ou_std:.04f}')
            for n in range(2):
                plt.plot([ou_mean_vals.min(), ou_mean_vals.max()],
                         [self._rate_limits[pop_name][n]] * 2, 'k--')
            #plt.get_current_fig_manager().window.showMaximized()
            plt.draw()
            plt.show()
            plt.savefig(dirpath_figs_pop / f'iter_{iter_num}.png')
