from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.data_proc import NetSpikesParams, NetRatesParams


@dataclass
class UCMapFitParams:

    pop_names: Tuple[str, ...] = ()

    # Parameters of the simulation result processing
    spikes_calc_params: NetSpikesParams = NetSpikesParams()  # sim_result -> spikes
    rates_calc_params: NetRatesParams = NetRatesParams(      # spikes -> rates
        time_limits=(0.5, None)  # time window for rate calculation (s)
    )

    # U-C mapping type and hyperparameters
    map_type: MapFuncType = MapFuncType.SIGMOID_1D
    map_params: Dict = field(default_factory=lambda: {
        'x_limits': (0, np.inf),  # intput is a rate, should be non-negative
        'y_limits': (0, np.inf)   # output is a rate, should be non-negative
    })

    # Fitting bounds for U-C mapping parameters
    fit_param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Range of the input values used for fitting
    inp_limits: Dict[str, Tuple[float, float]] | None = None

    # Weights for the fitting (prioritize the points with low rates)
    # Formula: weight = np.clip(rate ** fit_weight_pow, *fit_weight_limits)
    use_fit_weights: bool = False
    fit_weight_pow: float = 0
    fit_weight_limits: Tuple[float, float] = (0, np.inf)

    # Paramteres of the fitting algorithm (tolerances, etc...)
    map_fit_params: MapFitParams = MapFitParams()

    def __post_init__(self):
        if self.spikes_calc_params.pop_names not in (self.pop_names, None):
            raise ValueError('spikes_calc_params.pop_names should match pop_names')
        if self.rates_calc_params.pop_names not in (self.pop_names, None):
            raise ValueError('rates_calc_params.pop_names should match pop_names')
