import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # to avoid graphics error on servers

from netpyne.batchtools import comm, specs
from netpyne import sim

from create_base_cfg import create_base_cfg
from create_net_params import create_net_params
from load_module import load_module


# Folder names for experiment configs and results (relative to this script)
DIRNAME_EXP_CONFIGS = 'exp_configs'
DIRNAME_EXP_RESULTS = 'exp_results'

# Take batch flag from command line arguments, default to False
parser = argparse.ArgumentParser(description="Run experiment script.")
parser.add_argument('--batch', action='store_true',
                    help="Run in batch mode.")
parser.add_argument('--name', type=str,
                    help="Experiment name (required if not in batch mode).")
args, _ = parser.parse_known_args()
is_batch = args.batch

if not args.batch and args.name is None:
    raise ValueError("Either --name or --batch is requred")

need_run = True

# Experiment name (define the folder name in exp_configs and exp_results)
if not is_batch:
    #exp_name = 'test_run_2'
    exp_name = args.name


dirpath_self = Path(__file__).resolve().parent

if is_batch:
    # Preliminary: get cfg from batchtools to identify exp_name (simLabel),
    # which is then used to  generate the path to exp_cfg.py
    exp_name = specs.mappings['simLabel'][:-6]  # cut away job id

print(f'Experiment name: {exp_name}, batch={is_batch}')

# Import experiment-specific config and batch py-files
fpath_exp_cfg = dirpath_self / DIRNAME_EXP_CONFIGS / exp_name / 'exp_cfg.py'
fpath_exp_batch = dirpath_self / DIRNAME_EXP_CONFIGS / exp_name / 'batch_params.py'
#print(f'Experiment config: {fpath_exp_cfg}')
cfg_mod = load_module(fpath_exp_cfg)
if is_batch:
    batch_mod = load_module(fpath_exp_batch)

# Initialize config object, common for every experiment of the model
cfg = create_base_cfg()

# Apply experiment-specific config modifications
print(f'>>>> Before: {cfg.equal_inputs}')
cfg_mod.apply_exp_cfg(cfg)
print(f'>>>> After: {cfg.equal_inputs}')

# Automatically set the experiment name in config
if not is_batch:
    cfg.simLabel = exp_name
    cfg.saveFolder = str(dirpath_self / DIRNAME_EXP_RESULTS / exp_name)

# Update config by batchtools (if applicable)
cfg.update_cfg()

# Apply experiment-specific post-update config modifications
# (derive other params from the ones set by batchtools in update_cfg)
if is_batch and hasattr(batch_mod, 'post_update'):
    batch_mod.post_update(cfg)

# Create netParams based on the config
netParams = create_net_params(cfg)

comm.initialize()

# Save the config into the output folder
if comm.is_host():
    cfg.save("{}/{}_cfg.json".format(cfg.saveFolder, cfg.simLabel))
    netParams.save('{}/{}_netParams.json'.format(cfg.saveFolder, cfg.simLabel))

if need_run:
    
    # Prepare simulation
    sim.initialize(simConfig=cfg, netParams=netParams)
    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()            			# create connections between cells based on params
    sim.net.addStims() 							# add network stimulation
    
    # Run simulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
    sim.runSim()                      			# run parallel Neuron simulation  
    sim.gatherData()                  			# gather spiking data and cell info from each node
    
    # Save the results
    sim.cfg.savePickle = 1
    sim.saveData(include=['simConfig', 'netParams', 'simData', 'net'])
    
    # Plot the result
    sim.analysis.plotData()         			# plot spike raster etc

# Finalize
if comm.is_host():
   out_json = json.dumps({'loss': 0})
   comm.send(out_json)
   comm.close()
