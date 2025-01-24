import importlib.util
import json
import os
from pathlib import Path
import pickle as pkl
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid graphics error on servers

from netpyne.batchtools import comm, specs
from netpyne import sim
from netpyne.specs import NetParams

from create_base_cfg import create_base_cfg
from create_net_params import create_net_params


def _load_module(fpath_mod):
    mod_spec = importlib.util.spec_from_file_location(
        'module.name', fpath_mod)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules['module.name'] = mod
    mod_spec.loader.exec_module(mod)
    return mod


need_run = True


# Preliminary: get cfg from batchtools to identify exp_name (simLabel),
# which is then used to  generate the path to exp_cfg.py
exp_name = specs.mappings['simLabel'][:-6]  # cut away job id

dirpath_self = Path(__file__).resolve().parent

# Import experiment-specific config py-file
dirpath_exp = dirpath_self / 'exp_configs' / exp_name
fpath_exp_cfg = dirpath_exp / 'exp_cfg.py'
cfg_mod = _load_module(fpath_exp_cfg)

# Initialize config object, common for every experiment of the model
cfg = create_base_cfg()

# Apply experiment-specific config modifications
cfg_mod.apply_exp_cfg(cfg)

# Update config by batchtools (including cfg.simLabel and cfg.saveFolder)
cfg.update_cfg()

# Create netParams based on the config
netParams = create_net_params(cfg)

comm.initialize()

# Save cfg and netParams into the output folder
if comm.is_host():
    cfg.save('{}/{}_cfg.json'.format(cfg.saveFolder, cfg.simLabel))
    netParams.save('{}/{}_netParams.json'.format(cfg.saveFolder, cfg.simLabel))

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

# Close the communication with the batchtools master process
if comm.is_host():
   out_json = json.dumps({'loss': 0})
   comm.send(out_json)
   comm.close()
