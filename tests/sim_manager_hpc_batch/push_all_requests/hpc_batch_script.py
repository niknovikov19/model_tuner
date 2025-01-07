import json
import os
from pathlib import Path
import pickle
import sys
import time


def joinpath_hpc(base, *args):
    return Path(base).joinpath(*args).as_posix()


print('Test script started', flush=True)

dirpath_base = sys.argv[1]

fpath_reqs_json = joinpath_hpc(dirpath_base, 'requests/requests.json')
dirpath_res = joinpath_hpc(dirpath_base, 'results')

print(f'File with sim requests: {fpath_reqs_json}', flush=True)
print(f'Output folder for the results: {dirpath_res}', flush=True)

# Read the requests
with open(fpath_reqs_json, 'r') as fid:
    sim_requests = json.load(fid)

# Process the resuests
for sim_label, sim_params in sim_requests.items():
    print(f'Process request {sim_label}', flush=True)
    sim_params['label'] = sim_label
    sim_params['processed'] = True
    fpath_res = os.path.join(dirpath_res, f'{sim_label}.pkl')
    with open(fpath_res, 'wb') as fid:
        pickle.dump(sim_params, fid)
    print('Done', flush=True)
    time.sleep(2)

print('Test script finished')