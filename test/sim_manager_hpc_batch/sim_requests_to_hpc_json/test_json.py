from pathlib import Path
from pprint import pprint
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions

from sim_manager_hpc_batch import SimManagerHPCBatch
from ssh_client import SSHClient


def delete_file(fs, fpath):
    if fs.exists(fpath):
        print(f'Delete: {fpath}')
        fs.remove(fpath)
    if fs.exists(fpath):
        raise RuntimeError(f'Deleted file still exists: {fpath}')

# SSH parameters
hostname = 'lethe.downstate.edu'
username = 'niknovikov19'
port = 1415
private_key_path = r'C:\Users\aleks\.ssh\id_rsa_lethe'

# Local folder
dirpath_local = Path(__file__).resolve().parent

# HPC folders
dirpath_hpc_base = Path('/ddn/niknovikov19/test/model_tuner/test_sim_requests_to_json')
dirpath_hpc_req = dirpath_hpc_base / 'requests'

# Path to json file with sim requests (all, subset)
fpath_req_json = (dirpath_hpc_base / 'sim_requests.json').as_posix()
fpath_req_sub_json = (dirpath_hpc_base / 'sim_requests_sub.json').as_posix()

# SSH Client
ssh_client = SSHClient(hostname, username, port, private_key_path)

with ssh_client.get_filesys_handler() as fs:
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in [dirpath_hpc_base, dirpath_hpc_req]:
        fs.makedirs(dirpath.as_posix(), permissions=perm, recreate=True)
    
    # Delete remote files
    print('Delete old files...')
    for fpath in [fpath_req_json, fpath_req_sub_json]:
        delete_file(fs, fpath)

# Simulation manager
sim_manager = SimManagerHPCBatch(ssh_client, None)

# Add sim requests
sim_manager.add_sim_request('req1', {'name': 'Request 1', 'float_par': 1.1})
sim_manager.add_sim_request('req2', {'name': 'Request 2', 'float_par': 2.2})
sim_manager.add_sim_request('req3', {'name': 'Request 3', 'float_par': 3.3})

# Create json file with all requests
print('Creating json file (all requests)')
sim_manager._sim_requests_to_hpc_json(fpath_req_json)

# Create json file with some of the requests
print('Creating json file (req1, req3)')
sim_manager._sim_requests_to_hpc_json(fpath_req_sub_json, labels_used=['req1', 'req3'])

# Read the resulting json files
with ssh_client.get_filesys_handler() as fs:
    for fpath in [fpath_req_json, fpath_req_sub_json]:
        with fs.open(fpath, 'r') as fid:
            content = fid.read()
            print('\n-------------')
            print(f'Content of {fpath}:')
            pvprint(content)
