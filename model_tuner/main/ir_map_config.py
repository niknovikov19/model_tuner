from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.data_proc import NetSpikesParams, NetRatesParams


@dataclass
class IRMapFitParams:

    pop_names: Tuple[str, ...] = ()

    # Parameters of the batch experiment that probes a range of input values
    dirpath_batch: str = ''
    exp_name: str = ''
    batch_param_name: str = ''  # name of the parameter that varies in the batch

    # Parameters of the simulation result processing
    spikes_calc_params: NetSpikesParams = NetSpikesParams()  # sim_result -> spikes  ## time_limits is list
    rates_calc_params: NetRatesParams = NetRatesParams(      # spikes -> rates
        time_limits=(0.5, None)  # time window for rate calculation (s)
    )

    # I-R mapping type and hyperparameters
    map_type: MapFuncType = MapFuncType.RICHARDS_1D  # asymmetric sigmoid
    map_params: Dict | None = None  ## x_limits missing

    # Fitting bounds for I-R mapping parameters
    fit_param_bounds: Dict[str, Tuple[float, float]] | None = None ## defaulted

    # Range of the input values used for fitting
    inp_limits: Dict[str, Tuple[float, float]] | None = None

    # Weights for the fitting (prioritize the points with low rates)
    # Formula: weight = np.clip(rate ** fit_weight_pow, *fit_weight_limits)
    use_fit_weights: bool = True
    fit_weight_pow: float = 0.5
    fit_weight_limits: Tuple[float, float] = (0.1, 10)

    # Paramteres of the fitting algorithm (tolerances, etc...)
    map_fit_params: MapFitParams = MapFitParams()

    def __post_init__(self):
        if self.spikes_calc_params.pop_names not in (self.pop_names, None):
            raise ValueError('spikes_calc_params.pop_names should match pop_names')
        if self.rates_calc_params.pop_names not in (self.pop_names, None):
            raise ValueError('rates_calc_params.pop_names should match pop_names')
        self.map_params = self.map_params or {
            'y_limits': (0, np.inf)  # output is a rate, should be non-negative
        }
        self.fit_param_bounds = self.fit_param_bounds or {
            'q': (1, 20)  # asymmetry coefficient of RICHARDS_1D mapping
        }
