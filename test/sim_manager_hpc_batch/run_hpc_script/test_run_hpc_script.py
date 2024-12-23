from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions

from sim_manager_hpc_batch import SimManagerHPCBatch
from ssh_client import SSHParams, SSHClient


def delete_file(fs, fpath):
    if fs.exists(fpath):
        print(f'Delete: {fpath}')
        fs.remove(fpath)
    if fs.exists(fpath):
        raise RuntimeError(f'Deleted file still exists: {fpath}')

# SSH parameters
ssh_par = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)

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

# Log and result files
fpath_log_hpc = (dirpath_hpc_log / 'log.out').as_posix()
fpath_result_hpc = (dirpath_hpc_result / 'result.out').as_posix()

# SSH Client
with SSHClient(ssh_par) as ssh:

    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in [dirpath_hpc_base, dirpath_hpc_log, dirpath_hpc_result]:
        ssh.fs.makedirs(dirpath.as_posix(), permissions=perm, recreate=True)
    
    # Delete remote files: script, log, result
    print('Delete old files...')
    for fpath in [fpath_script_hpc, fpath_log_hpc, fpath_result_hpc]:
        delete_file(ssh.fs, fpath)
    
    # Upload the script to HPC
    ssh.fs.upload_file(fpath_script_local, fpath_script_hpc)   
    
    # Simulation manager
    print('Upload the script...')
    sim_manager = SimManagerHPCBatch(ssh, None)
    
    # Run the script, pass dirpath_hpc_base as a command line argument
    print('Run the script...')
    sim_manager._run_hpc_script(
        fpath_script_hpc, fpath_log_hpc, cmd_args=dirpath_hpc_base.as_posix()
    )
    print('Done')

    # Monitor the log until the result file appears
    while True:
        finished = ssh.fs.exists(fpath_result_hpc)
        with ssh.fs.open(fpath_log_hpc, 'r') as fid:
            content = fid.read()
            print('-------------')
            print('>>> Content of the log file:')
            print(content)
            time.sleep(0.2)
        if finished:
            break
    
    print('-------------')
    print('>>> Content of the result file:')
    with ssh.fs.open(fpath_result_hpc, 'r') as fid:
        content = fid.read()
        print(content)