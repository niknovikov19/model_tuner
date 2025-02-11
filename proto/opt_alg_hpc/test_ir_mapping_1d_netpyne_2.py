import logging
import os
from pathlib import Path
import pickle
from pprint import pprint
import sys
import time
from typing import List

#sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
from model_tuner.opt.map_funcs import MapFuncType

from proc_params import ProcStepParams, NetSpikesParams, NetRatesParams
from batch_metric_getter import BatchMetricGetter1D


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
        #plt.xlim(x.min(), x.max())
        #plt.ylim(y.min(), y.max())
        plt.xlim(*r_limits[pop_name])
        plt.ylim(0, 50)


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

r_limits = {
    'L2e': (0, 250),
    'L2i': (0, 1000),
    'L4e': (0, 250),
    'L4i': (0, 1000),
}
# =============================================================================
# r_limits = {
#     'L2e': (0, 500),
#     'L2i': (0, 2000),
#     'L4e': (0, 500),
#     'L4i': (0, 2000),
# }
# =============================================================================

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
    
    # Fit input-to-regime mapping for a pop
    pop_ir_mapper = PopIREmpiricalMapper1D(
        #map_type=MapFuncType.EXP_1D,
        #map_type=MapFuncType.SIGMOID_1D,
        map_type=MapFuncType.RICHARDS_1D,
        map_params = {
            'x_positive': True,
            'y_positive': True
        }
    )
    rlim = r_limits[pop_name]
    mask = (inp_rates >= rlim[0]) & (inp_rates <= rlim[1])
    pop_ir_mapper.fit_from_data(inp_rates[mask], pop_rates[pop_name][mask])
    
    net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)

need_plot = 1
if need_plot:
    plot_ir_mapping(bmg, batch_param_name, net_ir_mapper, pop_names, r_limits)
