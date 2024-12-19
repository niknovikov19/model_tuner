from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

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
dirpath_hpc_base = Path('/ddn/niknovikov19/test/model_tuner/test_run_hpc_script')
dirpath_hpc_log = dirpath_hpc_base / 'log'
dirpath_hpc_result = dirpath_hpc_base / 'result'

# Script to run
fname_script = 'hpc_script.py'
fpath_script_local = (dirpath_local / fname_script).as_posix()
fpath_script_hpc = (dirpath_hpc_base / fname_script).as_posix()

# Log file
fpath_log_hpc = (dirpath_hpc_log / 'log.out').as_posix()

# SSH Client
ssh_client = SSHClient(hostname, username, port, private_key_path)

with ssh_client.get_filesys_handler() as fs:
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in [dirpath_hpc_base, dirpath_hpc_log, dirpath_hpc_result]:
        fs.makedirs(dirpath.as_posix(), permissions=perm, recreate=True)
    
    # Delete remote files: script, log, result
    print('Delete old files...')
    for fpath in [fpath_script_hpc, fpath_log_hpc]:
        delete_file(fs, fpath)
    # ...
    
# Upload the script to HPC
ssh_client.upload_file(fpath_script_local, fpath_script_hpc)   

# Simulation manager
print('Upload the script...')
sim_manager = SimManagerHPCBatch(ssh_client, None)

# Run the script
print('Run the script...')
sim_manager._run_hpc_script(
    fpath_script_hpc, fpath_log_hpc, cmd_args='test_arg'
)
print('Done')

# Read the log several times with sleep intervals
with ssh_client.get_filesys_handler() as fs:
    print('Content of the log file:')
    for n in range(7):
        with fs.open(fpath_log_hpc, 'r') as fid:
            content = fid.read()
            print(f'---- {n} ----')
            print(content)
            time.sleep(0.2)
    print('-------------')
