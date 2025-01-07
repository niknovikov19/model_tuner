from pathlib import Path
import pickle
from pprint import pprint
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions

from ssh_client import SSHClient
from sim_manager import SimStatus
from sim_manager_hpc_batch import SimManagerHPCBatch, SimBatchPaths
from ssh_params import SSHParams


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
ssh_par_lethe = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)
ssh_par_grid = SSHParams(
    host='grid',
    user='niknovikov19',
    fpath_private_key=r'C:\Users\aleks\.ssh\id_ed25519_grid'
)

# Local folder
dirpath_local = str(Path(__file__).resolve().parent)

# HPC base folder
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_run_batch'

# HPC paths
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

# Simulation labels
num_req = 5
sim_labels = [f'req{n}' for n in range(num_req)]

# Scripts to run
scripts_info = {
    'batch': {'name': 'hpc_batch_script.py'},
    'job': {'name': 'hpc_job_script.py'}
}
for info in scripts_info.values():
    info['fpath_local'] = joinpath_local(dirpath_local, info['name'])
    info['fpath_hpc'] = joinpath_hpc(dirpath_hpc_base, info['name'])

with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    
    # Test grid connection
    print('Test grid connection:')
    result = ssh.conn.run('uname -a', hide=True)
    print(result.stdout.strip())
    
    # Simulation manager
    sim_manager = SimManagerHPCBatch(
        ssh=ssh,
        fpath_batch_script=scripts_info['batch']['fpath_hpc'],
        batch_paths=hpc_paths,
        conda_env='netpyne_batch'
    )
    
    # Result files
    fpath_res_hpc_lst = [sim_manager.get_sim_result_path(label)
                         for label in sim_labels]
    
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in hpc_paths.get_used_folders():
        ssh.fs.makedirs(dirpath, permissions=perm, recreate=True)
    
    # Delete remote files: scripts, log, results
    print('Delete old files...')
    fpaths_todel = (hpc_paths.get_all_files() + fpath_res_hpc_lst +
                    [info['fpath_hpc'] for info in scripts_info.values()])
    for fpath in fpaths_todel:
        delete_file(ssh.fs, fpath)
        
    # Upload the scripts (batch and job) to HPC
    print('Upload batch and job scripts...')
    for info in scripts_info.values():
        ssh.fs.upload_file(info['fpath_local'], info['fpath_hpc'])
    
    # Add sim requests
    for n, label in enumerate(sim_labels):
        sim_manager.add_sim_request(
            label, {'name': f'Request {n}', 'float_par': float(n)}
        )
    
    # Push the requests to HPC
    print('Push the simulation requests to HPC...')
    sim_manager.push_all_requests()
    print('Done')
    
    print('Waiting for completion', end='', flush=True)
    while not sim_manager.is_finished():
        print('.', end='', flush=True)
        time.sleep(0.5)
    print()
    pprint(sim_manager.get_all_sim_statuses())
    
    # Read the results
    print('\nResults:')
    for label in sim_labels:
        print(f'\n------ {label} ------')
        if sim_manager.get_sim_status(label) == SimStatus.DONE:
            fpath_res = sim_manager.get_sim_result_path(label)
            with ssh.fs.open(fpath_res, 'rb') as fid:
                content = pickle.load(fid)
            pprint(content)
        else:
            print('ERROR')
