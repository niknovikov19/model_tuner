import logging
import os
from pathlib import Path
import pickle
from pprint import pprint
import shutil
import sys
import time
import traceback
from typing import Dict

print(f'EXEC: {sys.executable}')

from fs.permissions import Permissions
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from model_tuner.ssh import SSHParams, SSHClient
from model_tuner.sim_manager import (
    SimManagerHPCBatchQsub,
    SimBatchPaths,
    SimResultLocator,
    SimStatus,  
    HPCJobSubmitParams
)

from model_tuner.data_proc import (
    DataKeeper
)

from model_tuner.main import (
    OptExperimentParamsBase
)

from model_tuner.utils import load_yaml

# Imports from the same follder as this script
from oumean_range_tuner import OUMeanRangeTuner 

# Needed for unpickling files that were created with the old folder structure
import sys
from model_tuner.data_proc import data_types, proc_params
sys.modules['data_types'] = data_types
sys.modules['proc_params'] = proc_params


def fs_delete(fs, path):
    if fs.exists(path):
        print(f'Delete: {path}')
        fs.remove(path)
    if fs.exists(path):
        raise RuntimeError(f'Deleted item still exists: {path}')

def joinpath_hpc(base, *args):
    return Path(base).joinpath(*args).as_posix()

def joinpath_local(base, *args):
    return str(Path(base).joinpath(*args))


run_on_hpc = 0

# Local base folder (configs in the root, results in subfolders)
if run_on_hpc:
    dirpath_base_local = Path(
        '/ddn/niknovikov19/repo/model_tuner/test_data/range_search')
else:
    dirpath_base_local = Path(
        r'D:\WORK\Salvador\repo\model_tuner\test_data\range_search')

# Experiment name
exp_name = 'exp_test_2'

# Max. number of iterations
# (don't put it to config, so it can be increased later)
n_iter = 50

# Action flags
need_delete_prev_results = 0
need_plot_iter = 1
need_plot_res = 1


#### Configs

# Local experiment folder
dirpath_exp_local = (
    dirpath_base_local / 'subnet_state1_mech1_nosub_wmult_0.1' / exp_name
)

exp_name_base = f'sim_manager_batch/exp_subnet_state1_mech1_nosub_wmult_0.1'

# Load config files
configs = {
    'ssh_params': {'class': None},
    'exp_params': {'class': OptExperimentParamsBase}
}
for config_name, config_info in configs.items():
    config_path = dirpath_exp_local / f'{config_name}.yaml'
    config_info['data'] = load_yaml(config_path, data_class=config_info['class'])

# SSH params
ssh_params = configs['ssh_params']['data']
ssh_par_lethe = SSHParams(**ssh_params['lethe'])
ssh_par_grid = SSHParams(**ssh_params['grid'])

# Experiment params
exp_params: OptExperimentParamsBase = configs['exp_params']['data']


#### Folders

# Local folder to store intermediate optimization plots
dirpath_figs_local = dirpath_exp_local / 'opt_figs'
os.makedirs(dirpath_figs_local, exist_ok=True)

# Local folder to store optimization info
dirpath_info = dirpath_exp_local / 'info'
os.makedirs(dirpath_info, exist_ok=True)

# HPC base folder for the experiment data
exp_name_hpc = exp_name.replace('=', '_').replace('(', '').replace(')', '')
dirpath_hpc_base = exp_params.dirpath_hpc_base + '/' + exp_name_hpc

# HPC paths used in batch simulations
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)


#### Other preparations

# Initialize DataKeeper
dirpath_dk = str(dirpath_exp_local / 'data_keeper')
os.makedirs(dirpath_dk, exist_ok=True)
dk = DataKeeper(dirpath_dk)

logging.basicConfig(level=logging.ERROR, force=True)

# Reconfigure SSH if running on HPC
if run_on_hpc:
    ssh_par_fs=ssh_par_grid
    ssh_par_conn=ssh_par_grid
else:
    ssh_par_fs=ssh_par_lethe
    ssh_par_conn=[ssh_par_lethe, ssh_par_grid]


#### Task-specific part

# Parameters
pop_names = ['PV2', 'PV3']
ou_std_vals = [0, 0.03]
num_ou_mean_vals = 10
ou_mean_range_start = (-0.1, 0.1)
rate_limits = (0.2, 30)
tolerance = 0.05
tcalc_win = (2, None)   # time window for rate calculation
final_run = False   # when the range is found, and we need
                    # the last run with regularly located points

# Range tuner object
range_tuner = OUMeanRangeTuner(
    pop_names, ou_std_vals, num_ou_mean_vals,
    ou_mean_range_start, rate_limits,
    tolerance, tcalc_win
    #dummy_mode=True
)


#### Main part
with SSHClient(ssh_par_fs=ssh_par_fs,
               ssh_par_conn=ssh_par_conn) as ssh:
    
    # Object that maps sim labels to sim result files
    sim_res_locator = SimResultLocator(hpc_paths.results_dir, ssh.fs)
    
    # Simulation manager
    sim_manager = SimManagerHPCBatchQsub(
        ssh=ssh,
        fpath_batch_script=exp_params.fpath_batch_script_hpc,
        batch_script_job_params=HPCJobSubmitParams(
            num_cores=8,
            memory='64G'
        ),
        batch_paths=hpc_paths,
        conda_env=exp_params.conda_env,
        job_cmdline_args=[exp_name_base],
        child_job_name=('j' + exp_name_base.split('/')[-1])[:10]
    )
    
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in hpc_paths.get_used_folders():
        ssh.fs.makedirs(dirpath, permissions=perm, recreate=True)
    
    # Delete remote files: scripts, log, results
    print('Delete old files...')
    paths_todel = []
    if need_delete_prev_results:
        paths_todel += ssh.fs.listdir(hpc_paths.results_dir)
    for path in paths_todel:
        fs_delete(ssh.fs, path)

    # Iterations of the main optimization algorithm
    iter_num = 1
    while iter_num < n_iter:
        print(f'==== Iter: {iter_num} ====')

        # Create simulation requests. Each request 
        # corresponds to an (ou_mean_num, ou_std_num) pair
        sim_requests = range_tuner.create_sim_requests(iter_num)
        
        # Loop over the requests and send them if needed
        for sim_label, sim_request in sim_requests.items():
            # Check if the simulation result already exists
            if sim_res_locator.result_exists(sim_label):
                print('Simulation result already exists, do not re-run')
                continue            
            # TODO: delete old result
            # Send the request via SimManager (non-blocking)          
            sim_manager.add_sim_request(sim_label, sim_request)
            print(f'Add request: {sim_request["input"]}')
        
        # Push simulation requests
        print('Push simulation requests to HPC', flush=True)
        sim_manager.push_all_requests()
        
        # Wait for simulation results
        print('Waiting for completion', end='', flush=True)
        while not sim_manager.is_finished():
            print('.', end='', flush=True)
            time.sleep(0.5)
        print('\nCompleted')
        
        pprint(sim_manager.get_all_sim_statuses())
        # TODO: check for error statuses
        
        # Check the existence of all simulation results
        results_ok = True
        Rc_lst = []
        for sim_label in sim_requests:
            if not sim_res_locator.result_exists(sim_label):
                print(f'No result found for {sim_label}', flush=True)
                sim_manager.remove_sim_request(sim_label)
                results_ok = False
        if not results_ok:
            print(f'NOT ALL RESULTS FOUND - RESTART THE ITERATION', flush=True)
            continue
        
        # Update ou_mean ranges based on the simulation results
        range_tuner.process_sim_results(dk, sim_res_locator, iter_num)

        # Save info about the current state of the optimization process
        fpath_info = dirpath_info / f'iter_rates_{iter_num}.pkl'
        with open(fpath_info, 'wb') as fid:
            pickle.dump(range_tuner._rates, fid)        
        
        # Visualize the iteration result
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            range_tuner.plot_rates(dirpath_figs_local, iter_num)

        if final_run:
            # Final run is done - exit
            print('DONE', flush=True)
            break
        elif range_tuner.is_done():
            # Ranges are found for all pops. - do the final run 
            print('ALL RANGES FOUND -> FINAL RUN', flush=True)
            range_tuner.final_step()
            final_run = True
        else:
            # Go to the next estimation of the ranges
            range_tuner.step()

        # Proceed to the next iteration
        iter_num += 1
