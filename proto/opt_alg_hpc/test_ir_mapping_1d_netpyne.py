from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.sim_manager import SimManagerHPCBatch, SimBatchPaths, SimStatus
from model_tuner.ssh import SSHParams, SSHClient

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
# NetUCMapper

from model_tuner.opt.map_funcs import MapFunc1D
from model_tuner.opt.map_funcs import MapFunc1DExp, MapFunc1DSigmoid

from proc_params import ProcStepParams, NetSpikesParams, NetRatesParams

from batch_metric_getter import BatchMetricGetter1D


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

inp_max = 1000

logging.basicConfig(level=logging.DEBUG, force=True)

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
        #map_type='exp_1d',
        map_type='sigmoid_1d',
        map_params = {
            'x_positive': False,
            'y_positive': True
        }
    )
    mask = (inp_rates <= inp_max)
    pop_ir_mapper.fit_from_data(inp_rates[mask], pop_rates[pop_name][mask])
    
    net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)


## Apply the resulting network I-R mapper to a range of input rates

pop_names = bmg.get_pop_names()

n_points = 200
rr_inp = np.linspace(0, inp_max, n_points)
rr_pop = {pop_name: np.zeros(n_points) for pop_name in pop_names}

for n, r_inp in enumerate(rr_inp):    
    # Combine pop inputs to a network input
    pop_inputs = {pop_name: PopInput1D(value=r_inp)
                  for pop_name in pop_names}
    net_input = NetInput1D(pop_inputs=pop_inputs)
    
    # Apply I-R mapping
    net_regime = net_ir_mapper.I_to_R(net_input)
    
    # Store the resulting pop rates
    for pop_name in pop_names:
        rr_pop[pop_name][n] = net_regime.pop_regimes[pop_name].value


## Visualize the results

logging.basicConfig(level=logging.ERROR, force=True)

plt.figure()

for n, pop_name in enumerate(pop_names):
    plt.subplot(1, len(pop_names), n + 1)
    
    # Fitted data produced by the I-R mapper
    plt.plot(rr_inp, rr_pop[pop_name])
    
    # Data used to "learn" the I-R mapping
    plt.plot(inp_rates, pop_rates[pop_name], 'k.')
    
    plt.title(pop_name)
    plt.xlabel('Input rate')
    plt.xlabel('Pop. rate')
    #plt.xlim(0, inp_max)
    plt.xlim(0, inp_max)
    plt.ylim(0, 40)
