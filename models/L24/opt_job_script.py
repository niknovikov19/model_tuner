import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid graphics error on servers

from netpyne.batchtools import comm, specs
from netpyne import sim
from netpyne.specs import NetParams

from create_base_cfg import create_base_cfg
from create_net_params import create_net_params

sys.path.append('/ddn/niknovikov19/repo/model_tuner')


def _load_module(fpath_mod):
    mod_spec = importlib.util.spec_from_file_location(
        'module.name', fpath_mod)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules['module.name'] = mod
    mod_spec.loader.exec_module(mod)
    return mod


# Initiate connection with the batchtools main process
comm.initialize()

# Import experiment-specific config py-file
dirpath_self = Path(__file__).resolve().parent
dirpath_exp = dirpath_self / 'exp_configs' / 'opt_batch'
fpath_exp_cfg = dirpath_exp / 'exp_cfg.py'
cfg_mod = _load_module(fpath_exp_cfg)

# Initialize config object, common for every experiment of the model
cfg = create_base_cfg()
cfg.sim_manager = {}

# Apply experiment-specific config modifications
cfg_mod.apply_exp_cfg(cfg)

# Update config by batchtools (including simLabel, saveFolder, 
# sim_manager.sim_label, sim_manager.requests_file)
cfg.update_cfg()

dirpath_res = cfg.saveFolder
sim_label = cfg.sim_manager['sim_label']
fpath_reqs = cfg.sim_manager['requests_file']

print(f'Job script started (label={sim_label})', flush=True)
print(f'File with sim requests: {fpath_reqs}', flush=True)
print(f'Output folder for the results: {dirpath_res}', flush=True)

# Set the sim label that was passed from SimManager
cfg.simLabel = sim_label

# Read the request
with open(fpath_reqs, 'r') as fid:
    sim_request = json.load(fid)[sim_label]

print('Request:')
print(sim_request)

# Get external input rates (for every pop) from the request
cfg.ext_input_rates = sim_request['input']
cfg.wmult = sim_request['wmult']

# Create netParams based on the config
netParams = create_net_params(cfg)

# Save cfg and netParams into the output folder
if comm.is_host():
    cfg.save('{}/{}_cfg.json'.format(cfg.saveFolder, cfg.simLabel))
    netParams.save('{}/{}_netParams.json'.format(cfg.saveFolder, cfg.simLabel))
    
need_run = 1
if need_run:
    
    # Prepare simulation
    sim.initialize(simConfig=cfg, netParams=netParams)
    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()            			# create connections between cells based on params
    sim.net.addStims() 							# add network stimulation
    
    # Run simulations
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
    sim.runSim()                      			# run parallel Neuron simulation  
    sim.gatherData()                  			# gather spiking data and cell info from each node
    
    # Save the results
    sim.cfg.savePickle = 1
    sim.saveData(include=['simConfig', 'netParams', 'simData', 'net'])
    
    # Plot the result
    sim.analysis.plotData()         			# plot spike raster etc

print('Job script finished')

# Close the communication with the batchtools main process
if comm.is_host():
   out_json = json.dumps({'loss': 0})
   comm.send(out_json)
   comm.close()
