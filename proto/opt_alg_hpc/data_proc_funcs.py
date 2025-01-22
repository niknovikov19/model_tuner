import logging
from typing import Any

#import numpy as np

from proc_params import NetSpikesParams, NetRatesParams, ProcParamsChain
from data_types import DataType, NetSpikesData, NetRatesData

import data_proc_utils as proc_utils
import netpyne_res_parse_utils as parse_utils


def extract_net_spikes(
        sim_res_data: Any,  # NetPyNE output
        par: NetSpikesParams,
        params_chain_in: ProcParamsChain = ProcParamsChain()
        ) -> NetSpikesData:
    
    logging.debug('data_proc_funcs.extract_net_spikes()')
    
    # Extract spikes from netpyne simulation result    
    S = parse_utils.get_net_spikes(
        sim_res_data,
        pop_names=par.pop_names,
        combine_cells=par.combine_cells,
        t0=par.time_limits[0],
        tmax=par.time_limits[1],
        subtract_t0=par.subtract_t0,
        ms=par.ms,
        ndigits=par.ndigits
    )
    
    # Get time limits of the spike data
    T = parse_utils.get_sim_duration(sim_res_data)
    t0 = par.time_limits[0]
    tmax = par.time_limits[1] or T
    if par.subtract_t0:
        tmax -= t0
        t0 = 0
    if par.ms:
        t0 *= 1000
        tmax *= 1000
        
    # Number of neurons
    net_size = parse_utils.get_net_size(sim_res_data)
    
    params_chain_out = params_chain_in.add_step('net_spikes', par)
    
    return NetSpikesData(
        data=S,
        time_limits=(t0, tmax),
        params=par,
        params_chain=params_chain_out,
        net_size=net_size
    )

def calc_net_rates(
        net_spikes: NetSpikesData,
        par: NetRatesParams,
        data_name: str = 'net_rates'
        ) -> NetRatesData:
    
    logging.debug('data_proc_funcs.calc_net_rates()')
    
    time_limits = par.time_limits
    if not time_limits[1]:
        time_limits = (time_limits[0], net_spikes.time_limits[1])  # resolve None
    
    # Call a low-level function
    net_rates = proc_utils.calc_net_rates(
        net_spikes=net_spikes.data,
        time_limits=time_limits,
        ncells=net_spikes.net_size,
        pop_names=par.pop_names
    )
    
    params_chain_out = net_spikes.params_chain.add_step(data_name, par)
    
    return NetRatesData(
        data=net_rates,
        data_name=data_name,
        params=par,
        params_chain=params_chain_out
    )
