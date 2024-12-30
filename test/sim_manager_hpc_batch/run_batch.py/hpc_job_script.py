import json
import os
import pickle
import sys
import time

from netpyne.batchtools import comm, specs


comm.initialize()

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
    sim_requests = json.load(fid)
sim_params = sim_requests[sim_label]

# Process the request
print(f'Process request {sim_label}', flush=True)
sim_params['label'] = sim_label
sim_params['processed'] = True
fpath_res = os.path.join(dirpath_res, f'{sim_label}.pkl')
with open(fpath_res, 'wb') as fid:
    pickle.dump(sim_params, fid)

print('Test script finished')

# Close the communication with the batchtools master process
if comm.is_host():
   out_json = json.dumps({'loss': 0})
   comm.send(out_json)
   comm.close()
