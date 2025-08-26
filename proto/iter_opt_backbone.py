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

from model_tuner.data_proc import DataKeeper

from model_tuner.main import (
    OptExperimentParams,
    get_sim_rates,
    plot_opt_iteration_pop,
    plot_ir_mapping_1d_slice
)

from model_tuner.utils import load_yaml

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

def set_qt_backend():
    global plt
    if 'matplotlib.pyplot' in sys.modules:
        del sys.modules['matplotlib.pyplot']  # remove old pyplot with bad backend
    matplotlib.use('Qt5Agg', force=True)  # re-set the backend
    import matplotlib.pyplot as plt_
    plt = plt_


run_on_hpc = 0

# Local base folder (configs in the root, results in subfolders)
if run_on_hpc:
    dirpath_base_local = Path(
        '/ddn/niknovikov19/repo/model_tuner/test_data/main/test_opt_A1_hpc_batch_qsub')
else:
    dirpath_base_local = Path(
        r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_opt_A1_hpc_batch_qsub')

# Experiment name
exp_name = 'test_5_pfr=(0.4_1.0_4)_wmult=0.02_alpha=0.2_autosz_sigline'

# Number of iterations
# (don't put it to config, so it can be increased later)
n_iter = 50

# Action flags
need_delete_prev_results = 0
need_plot_iter = 1
need_plot_res = 1


# Local experiment folder
dirpath_exp_local = dirpath_base_local / 'experiments' / exp_name

# Load config files
configs = {
    'ssh_params': {'class': None},
    'exp_params': {'class': OptExperimentParams}
}
for config_name, config_info in configs.items():
    config_path = dirpath_exp_local / f'{config_name}.yaml'
    config_info['data'] = load_yaml(config_path, data_class=config_info['class'])

# SSH params
ssh_params = configs['ssh_params']['data']
ssh_par_lethe = SSHParams(**ssh_params['lethe'])
ssh_par_grid = SSHParams(**ssh_params['grid'])

# Experiment params
exp_params: OptExperimentParams = configs['exp_params']['data']

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

# Main part
with SSHClient(ssh_par_fs=ssh_par_fs, ssh_par_conn=ssh_par_conn) as ssh:
    
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
        conda_env=exp_params.conda_env
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
        
        # Rc0 -> Ru
        Ru_lst = []
        #Ru_lst = uc_optimizer.suggest_Ru()

        # Labels of the simulations that will be added to the current batch
        sim_labels = []
        
        # Loop over the target regimes (pfr)
        for n, Ru in enumerate(Ru_lst):

            point_num = n
            print(f'Point: {point_num}')
            
            # Generate a unique simulation label
            sim_label = f'req_{iter_num}_{point_num}'
            sim_labels.append(sim_label)
            
            # Check if the simulation result already exists
            if sim_res_locator.result_exists(sim_label):
                print('Simulation result already exists, do not re-run')
                continue
            
            # TODO: delete old result
            
            # Calculate an input Iu that provides the unconnected regime Ru
            #Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_request = {
                #'input': Iu.to_values_dict(),
                #'connected': exp_params.model_cfg['connected'],
                #'wmult': exp_params.model_cfg['wmult']
            }            
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
        
        # Extract network regimes from simulation results
        print('Retrieving the results', end='', flush=True)
        results_ok = True
        Rc_lst = []
        for n, sim_label in enumerate(sim_labels):
            print('.', end='', flush=True)
            if sim_res_locator.result_exists(sim_label):
                sim_result_desc = sim_res_locator.locate_result(sim_label)            
                Rc = get_sim_rates(
                    dk, sim_res_locator, sim_label
                    #uc_map_params.spikes_calc_params,
                    #uc_map_params.rates_calc_params
                )
                Rc_lst.append(Rc)
            else:
                print(f'No result found for {sim_label}', flush=True)
                sim_manager.remove_sim_request(sim_label)
                results_ok = False

        if not results_ok:
            print(f'NOT ALL RESULTS FOUND - RESTART THE ITERATION', flush=True)
            continue
        print('\nCompleted', flush=True)
        
        # Fit UC mapping
        try:
            #uc_optimizer.fit_uc_mapper()
            uc_fit_ok = True
        except Exception as e:
            print(e)
            traceback.print_exc()
            uc_fit_ok = False

        # Save info about the current state of the optimization process
        cc = {'iter': slice(0, iter_num + 1)}
        info = {
            'pop_names': exp_params.pop_names,
        }
        fpath_info = dirpath_info / f'Ru_Rc_{sim_label}.pkl'
        with open(fpath_info, 'wb') as fid:
            pickle.dump(info, fid)        
        
        # Visualize the iteration result
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            set_qt_backend()
            plt.ion()
            
            dirpath_figs_iter = dirpath_figs_local / f'iter_{iter_num}'
            os.makedirs(dirpath_figs_iter, exist_ok=True)
            
            for pop_name in exp_params.pop_names:
                plt.figure(111)
                plt.clf()
                # ...
                plt.get_current_fig_manager().window.showMaximized()
                plt.draw()
                plt.show()
                plt.savefig(dirpath_figs_iter / f'{pop_name}.png')

        if not uc_fit_ok:
            print('U-C map fitting failed - stop')
            break

        # Proceed to the next iteration
        #uc_optimizer.next_iter()
        iter_num += 1
