from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions

from ssh_client import SSHClient
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
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_run_hpc_script'

# Script to run
fname_script = 'hpc_script.py'
fpath_script_local = joinpath_local(dirpath_local, fname_script)
fpath_script_hpc = joinpath_hpc(dirpath_hpc_base, fname_script)

# HPC paths used by SimManager
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

# Path to the result file
fpath_res_hpc = joinpath_hpc(hpc_paths.results_dir, 'result.out')

# SSH Client
with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    
    # Test grid connection
    print('Test grid connection:')
    result = ssh.conn.run('uname -a', hide=True)
    print(result.stdout.strip())

    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in hpc_paths.get_used_folders():
        ssh.fs.makedirs(dirpath, permissions=perm, recreate=True)
    
    # Delete remote files: script, log, result
    print('Delete old files...')
    fpaths_todel = (hpc_paths.get_all_files() + 
                    [fpath_res_hpc, fpath_script_hpc])
    for fpath in fpaths_todel:
        delete_file(ssh.fs, fpath)
    
    # Upload the script to HPC
    ssh.fs.upload_file(fpath_script_local, fpath_script_hpc)   
    
    # Simulation manager
    print('Upload the script...')
    sim_manager = SimManagerHPCBatch(
        ssh=ssh,
        fpath_batch_script=None,
        batch_paths=hpc_paths,
        conda_env='netpyne_batch'
    )
    
    # Run the script, pass dirpath_hpc_base as a command line argument
    print('Run the script...')
    sim_manager._run_hpc_script(
        fpath_script=fpath_script_hpc,
        fpath_log = hpc_paths.log_file,
        cmd_args=fpath_res_hpc
    )
    print('Done')

    # Monitor the log until the result file appears
    while True:
        finished = ssh.fs.exists(fpath_res_hpc)
        with ssh.fs.open(hpc_paths.log_file, 'r') as fid:
            content = fid.read()
            print('-------------')
            print('>>> Content of the log file:')
            print(content)
            time.sleep(0.2)
        if finished:
            break
    
    print('-------------')
    print('>>> Content of the result file:')
    with ssh.fs.open(fpath_res_hpc, 'r') as fid:
        content = fid.read()
        print(content)
