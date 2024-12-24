from pathlib import Path
import pickle
from pprint import pprint
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions

from sim_manager_hpc_batch import SimManagerHPCBatch, SimBatchPaths
from ssh_client import SSHParams, SSHClient


def delete_file(fs, fpath):
    if fs.exists(fpath):
        print(f'Delete: {fpath}')
        fs.remove(fpath)
    if fs.exists(fpath):
        raise RuntimeError(f'Deleted file still exists: {fpath}')

def joinpath_hpc(base, *args):
    return Path(base).joinpath(*args).as_posix()

def joinpath_local(base, *args):
    return str(Path(base).joinpath(*args))


# SSH parameters
ssh_par = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)

# Local folder
dirpath_local = Path(__file__).resolve().parent

# HPC base folder
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_push_all_requests'

# Simulation labels
num_req = 5
sim_labels = [f'req{n}' for n in range(num_req)]

# Script to run
fname_script = 'hpc_batch_script.py'
fpath_script_local = joinpath_local(dirpath_local, fname_script)
fpath_script_hpc = joinpath_hpc(dirpath_hpc_base, fname_script)

# HPC paths
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

with SSHClient(ssh_par) as ssh:
    # Simulation manager
    sim_manager = SimManagerHPCBatch(
        ssh,
        fpath_batch_script=fpath_script_hpc,
        batch_paths=hpc_paths
    )
    
    # Result files
    fpath_res_hpc_lst = [sim_manager.get_sim_result_path(label)
                         for label in sim_labels]
    
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in hpc_paths.get_used_folders():
        ssh.fs.makedirs(dirpath, permissions=perm, recreate=True)
    
    # Delete remote files: script, log, result
    print('Delete old files...')
    fpaths_todel = hpc_paths.get_all_files() + fpath_res_hpc_lst
    for fpath in fpaths_todel:
        delete_file(ssh.fs, fpath)
        
    # Upload the script to HPC
    print('Upload the script...')
    ssh.fs.upload_file(fpath_script_local, fpath_script_hpc)   
    
    # Add sim requests
    for n, label in enumerate(sim_labels):
        sim_manager.add_sim_request(
            label, {'name': f'Request {n}', 'float_par': float(n)}
        )
    
    # Push the requests to HPC
    print('Push the simulation requests to HPC...')
    sim_manager.push_all_requests()
    print('Done')
    
    while not sim_manager.is_finished():
        pprint(sim_manager.get_all_sim_statuses(update=False))
        time.sleep(0.2)
    pprint(sim_manager.get_all_sim_statuses(update=False))
    
    # Read the results
    print('\nResults:')
    for n, fpath in enumerate(fpath_res_hpc_lst):        
        with ssh.fs.open(fpath, 'rb') as fid:
            content = pickle.load(fid)
            print(f'\n------{n}------')
            pprint(content)
