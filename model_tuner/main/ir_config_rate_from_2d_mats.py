from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D


@dataclass
class IRMapConfigRateFrom2DRateCVMats:

    pop_names: Tuple[str, ...] = ()

    # Path to the file with the matrices of firing rates and CVs
    fpath_mats: str = ''

    # Parameters that were varied in the batch experiment
    batch_param_names: Tuple[str, str] = ('', '')

    # Batch parameter that parametrizes 1-d slice of 2-d matrix
    batch_param_main: str = ''

    # Secondary batch parameter (the other one is batch_param_main)
    @property
    def batch_param_sec(self) -> str:
        return next(
            (param for param in self.batch_param_names 
            if param != self.batch_param_main),
            None
        )

    # Method of slicing the firing rate matrix to get 1-d data
    # 'batch_param_ratio' - slice along the line with a fixed ratio
    #    of the batch parameters (batch_param_sec / batch_param_main)
    slice_method: Literal['batch_param_ratio'] = 'batch_param_ratio'

    # Ratio of the secondary batch parameter to the main one
    batch_param_ratio: float = 0.4

    # I-R mapping type and hyperparameters
    map_type: MapFuncType = MapFuncType.RICHARDS_1D  # asymmetric sigmoid
    map_params: Dict = field(default_factory=lambda: {
        'y_limits': (0, np.inf)  # output is a rate, should be non-negative
    })

    # Fitting bounds for I-R mapping parameters
    fit_param_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'q': (1, 20)  # asymmetry coefficient of RICHARDS_1D mapping
        }
    )

    # Range of the input values (batch_param_main) used for fitting
    inp_limits: Dict[str, Tuple[float, float]] | None = None

    # Weights for the fitting (prioritize the points with low rates)
    # Formula: weight = np.clip(rate ** fit_weight_pow, *fit_weight_limits)
    use_fit_weights: bool = True
    fit_weight_pow: float = 0.5
    fit_weight_limits: Tuple[float, float] = (0.1, 10)

    # Paramteres of the fitting algorithm (tolerances, etc...)
    map_fit_params: MapFitParams = MapFitParams()

    def __post_init__(self):
        if self.batch_param_main not in self.batch_param_names:
            raise ValueError(
                'batch_param_main should be one of batch_param_names'
            )
    
    def init_ir_mapper(self, need_plot: bool = True) -> NetIREmpiricalMapper1D:
        return _init_ir_mapper(self, need_plot)


def _init_ir_mapper(
        par: IRMapConfigRateFrom2DRateCVMats,
        need_plot: bool = True
        ) -> NetIREmpiricalMapper1D:
    pass

"""
    proc_params = {
        'net_spikes': par.spikes_calc_params,
        'net_rates': par.rates_calc_params
    }
    
    # Object that exctracts firing rates from batch sim results
    bmg = BatchMetricGetter1D(
        par.dirpath_batch,
        par.exp_name,
        par.pop_names,
        par.batch_param_name,
        proc_params
    )

    # Batch parameter values that characterize the model input
    inp_rates = bmg.get_batch_par_values(par.batch_param_name)

    pop_rates = {}

    # Network input-to-regime mapper
    net_ir_mapper = NetIREmpiricalMapper1D()

    for pop_name in bmg.get_pop_names():
        # Request firing rates of a pop (for every batch parameter value)
        pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
        
        # Select data points used for fitting
        mask = ((inp_rates >= par.inp_limits[pop_name][0]) & 
                (inp_rates <= par.inp_limits[pop_name][1]))
        xx = inp_rates[mask]
        yy = pop_rates[pop_name][mask]
        
        # Weights for fitting (prioritize the points with low rates)
        ww = None
        if par.use_fit_weights:
            ww = np.clip(
                yy ** par.fit_weight_pow, *par.fit_weight_limits
            )
        
        # Fit input-to-regime mapping for a pop
        pop_ir_mapper = PopIREmpiricalMapper1D(par.map_type, par.map_params)
        pop_ir_mapper.fit_from_data(
            xx, yy, fit_params=par.map_fit_params, weights=ww,
            bounds=par.fit_param_bounds
        )
        
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)

    if need_plot:
        _plot_ir_mapping(bmg, par.batch_param_name, net_ir_mapper)
    
    return net_ir_mapper


def _plot_ir_mapping(
        bmg: IRMapConfigRateFrom2DRateCVMats,
        batch_param_names: Tuple[str, str],
        net_ir_mapper: NetIREmpiricalMapper1D
        ):
    
    pop_names = bmg.get_pop_names()

    # Inputs and outputs used for fitting
    inp_rates = bmg.get_batch_par_values(batch_param_name)
    pop_rates = {}
    for pop_name in pop_names:
        pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
        
    r_limits = {
        'L2e': (0, 250),
        'L2i': (100, 750),
        'L4e': (0, 250),
        'L4i': (100, 750),
    }

    # Apply I-R mapping to a range of input rates
    n_points = 100
    rr_inp, rr_pop = {}, {}
    for pop_name in pop_names:
        rlim = r_limits[pop_name]
        rr_inp[pop_name] = np.linspace(rlim[0], rlim[1], n_points)
        rr_pop[pop_name] = np.zeros(n_points)
        for n, r_inp in enumerate(rr_inp[pop_name]):
            pop_inputs = {pop_name_: PopInput1D(value=r_inp)
                          for pop_name_ in pop_names}
            net_input = NetInput1D(pop_inputs=pop_inputs)
            net_regime = net_ir_mapper.I_to_R(net_input)        
            rr_pop[pop_name][n] = net_regime.pop_regimes[pop_name].value
    
    plt.figure()    
    for n, pop_name in enumerate(pop_names):
        plt.subplot(1, len(pop_names), n + 1)
        
        # Fitted data produced by the I-R mapper
        x, y = rr_inp[pop_name], rr_pop[pop_name]
        plt.plot(x, y)
        
        # Data used to "learn" the I-R mapping
        plt.plot(inp_rates, pop_rates[pop_name], 'k.')
        
        plt.title(pop_name)
        plt.xlabel('Input rate')
        plt.ylabel('Pop. rate')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
"""