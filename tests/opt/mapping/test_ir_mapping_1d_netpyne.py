import os
from typing import List
import sys

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
from model_tuner.opt.map_funcs import MapFuncType, MapFitParams

from model_tuner.data_proc import NetSpikesParams, NetRatesParams
from model_tuner.data_proc import BatchMetricGetter1D

# Needed for unpickling files that were created with the old folder structure
from model_tuner.data_proc import data_types, proc_params
sys.modules['data_types'] = data_types
sys.modules['proc_params'] = proc_params


def plot_ir_mapping(
        bmg: BatchMetricGetter1D,
        batch_param_name: str,
        net_ir_mapper: NetIREmpiricalMapper1D,
        pop_names: List[str],
        r_limits
        ):

    # Inputs and outputs used for fitting
    inp_rates = bmg.get_batch_par_values(batch_param_name)
    pop_rates = {}
    for pop_name in bmg.get_pop_names():
        pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)

    # Apply I-R mapping to a range of input rates
    n_points = 100
    rr_inp, rr_pop = {}, {}
    for pop_name in bmg.get_pop_names():
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
        plt.xlim(*r_limits[pop_name])
        plt.ylim(0, 50)
    plt.show()


dirpath_batch = (
    r'D:\WORK\Salvador\repo\model_tuner\models\L24\exp_results\rx_batch_unconn_2'
)
exp_name = 'rx_batch_unconn_2'
pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
batch_param_name = 'ext_inp_rate_common'

proc_params = {
    'net_spikes': NetSpikesParams(pop_names=pop_names),
    'net_rates': NetRatesParams(time_limits=(0.5, None))
}

# Limits for the input rates used for fitting
r_limits = {
    'L2e': (0, 250),
    'L2i': (0, 1000),
    'L4e': (0, 250),
    'L4i': (0, 1000),
}

# Fitting bounds for I-R mapping parameters
fit_param_bounds = {
    'q': (1, 10)
}

# Prameters of the formula that determines the weights for fitting
fit_weight_pow = 0.5
fit_weight_limits = (0.1, 10)

# Paramteres of the fitting algorithm
fit_params = MapFitParams(
    #return_first_guess=True
)

# Object that exctracts firing rates from batch sim results
bmg = BatchMetricGetter1D(
    dirpath_batch, exp_name, pop_names, batch_param_name, proc_params
)

# Batch parameter values that characterize the model input
inp_rates = bmg.get_batch_par_values(batch_param_name)

pop_rates = {}

# Network input-to-regime mapper
net_ir_mapper = NetIREmpiricalMapper1D()

for pop_name in bmg.get_pop_names():

    # Request firing rates of a pop (for every batch parameter value)
    pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)

    # Select data points used for fitting
    rlim = r_limits[pop_name]
    mask = (inp_rates >= rlim[0]) & (inp_rates <= rlim[1])
    xx = inp_rates[mask]
    yy = pop_rates[pop_name][mask]

    # Weights for fitting (prioritize the points with low rates)
    weights = np.clip(
        yy ** fit_weight_pow, fit_weight_limits[0], fit_weight_limits[1]
    )
    
    # Fit input-to-regime mapping for a pop
    pop_ir_mapper = PopIREmpiricalMapper1D(
        map_type=MapFuncType.RICHARDS_1D,
        map_params = {
            'x_limits': (0, np.inf),
            'y_limits': (0, np.inf)
        }
    )
    pop_ir_mapper.fit_from_data(
        xx, yy, fit_params=fit_params, weights=weights,
        bounds=fit_param_bounds
    )
    
    net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)

need_plot = 1
if need_plot:
    plot_ir_mapping(bmg, batch_param_name, net_ir_mapper, pop_names, r_limits)
