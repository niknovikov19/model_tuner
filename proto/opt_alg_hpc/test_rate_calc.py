import logging
from pathlib import Path
import pickle
from pprint import pprint

# Low-level data-processing
import netpyne_res_parse_utils as parse_utils
import data_proc_utils as proc_utils

# High-level data-processing
from proc_params import NetSpikesParams, NetRatesParams
from data_types import DataType, NetSpikesData, NetRatesData
import data_proc_funcs as proc_funcs

# High-level data-processing with DataKeeper as a storage
from sim_result import SimResultFile
from data_keeper import DataKeeper
from netpyne_result_parser import SimResultParserNetPyNE
from sim_data_proc import DataProcessor


logging.basicConfig(level=logging.DEBUG, force=True)

dirpath_root = Path(r'D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc'
                    r'\data\test_rate_calc')
fpath_sim_res = dirpath_root / 'sim_res_data.pkl'

pop_names = ['L2e', 'L2i', 'L4e', 'L4i']

test_low_level = 0
test_high_level = 0
test_dk = 1


#### Direct calculation with low-level functions
if test_low_level:
    print('\nDirect calculation with low-level functions')
    
    # Load sim result
    with open(fpath_sim_res, 'rb') as fid:
        sim_res = pickle.load(fid)
    
    # Extract spikes
    spikes = parse_utils.get_net_spikes(sim_res)
    
    # Sim time range and pops' sizes8
    time_limits = (0, parse_utils.get_sim_duration(sim_res))
    net_size = parse_utils.get_net_size(sim_res)
    
    # Compute rates from spikes
    rates = proc_utils.calc_net_rates(spikes, time_limits, net_size, pop_names)
    pprint(rates)


#### Direct calculation with high-level functions
if test_high_level:
    print('\nDirect calculation with high-level functions')

    # Load sim result
    with open(fpath_sim_res, 'rb') as fid:
        sim_res = pickle.load(fid)
    
    # Extract spikes
    spikes_par = NetSpikesParams(pop_names=pop_names)
    spikes_obj = proc_funcs.extract_net_spikes(sim_res, spikes_par)
    
    # Compute rates from spikes
    rates_par = NetRatesParams()
    rates_obj = proc_funcs.calc_net_rates(spikes_obj, rates_par)
    pprint(rates_obj.data)


#### Calculation with storage of results in DataKeeper
if test_dk:
    print('\nCalculation with storage of results in DataKeeper', flush=True)
    
    # Initialize DataKeeper
    dirpath_dk = str(dirpath_root / 'data_keeper')
    dk = DataKeeper(dirpath_dk)
    
    # Open sim result for parsing
    sim_res_desc = SimResultFile(fpath_sim_res)
    res_parser = SimResultParserNetPyNE(
        dk=dk,
        result_desc=sim_res_desc,
        defer_load=True
    )
    
    # Extract spikes from the sim result and strore them into dk
    spikes_par = NetSpikesParams(pop_names=pop_names)
    spikes_id = res_parser.extract_net_spikes(spikes_par)
    
    # Initialize data processor
    data_proc = DataProcessor(dk)
    
    # Load spikes from dk, calculate rates from them, save rates into dk
    rates_par = NetRatesParams()
    rates_id = data_proc.calc_net_rates(spikes_id, rates_par)
    
    # Load rates from dk
    rates_dk = data_proc.load_data(rates_id)
    pprint(rates_dk.data)
    
    
