import os
from typing import List
import sys

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D

from model_tuner.data_proc import BatchMetricGetter1D

from model_tuner.main import IRMapConfigRateFrom1DSim
from model_tuner.utils import load_yaml

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


dirpath_base = r'D:\WORK\Salvador\repo\model_tuner\test_data\test_ir_mapping_1d_netpyne'
os.makedirs(dirpath_base, exist_ok=True)
fpath_yaml = os.path.join(dirpath_base, 'config.yaml')

# Load from YAML
par = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom1DSim)

proc_params = {
    'net_spikes': par.spikes_calc_params,
    'net_rates': par.rates_calc_params
}

# Object that exctracts firing rates from batch sim results
bmg = BatchMetricGetter1D(
    par.dirpath_batch, par.exp_name, par.pop_names,
    par.batch_param_name, proc_params
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

need_plot = 1
if need_plot:
    plot_ir_mapping(
        bmg, par.batch_param_name, net_ir_mapper,
        par.pop_names, par.inp_limits
    )
    input('Press any key...')
