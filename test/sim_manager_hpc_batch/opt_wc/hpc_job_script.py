import json
import os
import pickle
import sys
import time

from netpyne.batchtools import comm, specs

sys.path.append('/ddn/niknovikov19/repo/model_tuner')
from defs_wc import NetInputWC, NetRegimeWC 
from model_wc import ModelDescWC, run_wc_model


# Initiate connection with the batchtools main process
comm.initialize()

# Get params specific for this job from the batchtools main process (via cfg)
cfg = specs.SimConfig()
cfg.sim_manager = {}
cfg.update_cfg()

dirpath_res = cfg.saveFolder
sim_label = cfg.sim_manager['sim_label']
fpath_reqs = cfg.sim_manager['requests_file']

print(f'Job script started (label={sim_label})', flush=True)
print(f'File with sim requests: {fpath_reqs}', flush=True)
print(f'Output folder for the results: {dirpath_res}', flush=True)

# Read the request
with open(fpath_reqs, 'r') as fid:
    sim_request = json.load(fid)[sim_label]

print('Request:')
print(sim_request)

# Extract the data required for simulation from the request
model = ModelDescWC(**sim_request['wc_model'])
Iext = NetInputWC(**sim_request['Iext'])
R0 = NetRegimeWC(**sim_request['R0'])
sim_par = sim_request['wc_sim_params']

# Run simulation
print(f'Process request {sim_label}', flush=True)
R, sim_info = run_wc_model(
    model, Iext, sim_par['niter'], R0, sim_par['dr_mult']
)

# Store the result
sim_result = {'R': R, 'sim_info': sim_info}
fpath_res = os.path.join(dirpath_res, f'{sim_label}.pkl')
with open(fpath_res, 'wb') as fid:
    pickle.dump(sim_result, fid)

print('Job script finished')

# Close the communication with the batchtools main process
if comm.is_host():
   out_json = json.dumps({'loss': 0})
   comm.send(out_json)
   comm.close()
