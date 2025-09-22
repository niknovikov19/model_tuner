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

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import NetIRMapper1DInterp
from model_tuner.opt.uc_mappers import NetUCMapper1D

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
    IRMapConfigRateInterpFrom2DXR,
    UCMapFitParams,
    OptExperimentParams,
    get_sim_rates,
    plot_opt_iteration_pop,
    plot_ir_mapping_1d_slice
)

from model_tuner.utils import load_yaml, set_qt_backend

from model_tuner.opt.uc_optimizer import (
    OptStrategyParams, UCOptimizer
)

from plot_ir_ougrid_interp_ import plot_ir_ougrid_interp

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


#### Experiment parameters

run_on_hpc = 0

# Local base folder
if run_on_hpc:
    raise NotImplementedError('Set local base folder for HPC')
else:
    dirpath_base_local = Path(
        r'D:\WORK\Salvador\repo\model_tuner\opt_experiments\ou_surr_to_full\data')

# Experiment group name
exp_name_base = 'exp_subnet_state1_mech1_nosub_wmult_0.1'
exp_name_base_hpc = 'sim_manager_batch/' + exp_name_base

# Experiment name
exp_name = 'L4_thal_conn_5s_spline_alpha_0.1_auto_pfr_2'

# Local experiment folder
dirpath_exp_local_base = dirpath_base_local / exp_name_base
dirpath_exp_local = dirpath_exp_local_base / exp_name

# Name of the target state (corresponds to csv file with target rates)
target_name = 'target_state_1'

# Number of iterations
# (don't put it to config, so it can be increased later)
n_iter = 25

# Action flags
need_delete_prev_results = 0
need_plot_ir = 0
need_plot_iter = 1
need_plot_iter_ir = 0
need_plot_res = 1


#### Configs

# Load config files
configs = {
    'ir_map_params': {'class': IRMapConfigRateInterpFrom2DXR},
    'uc_map_params': {'class': UCMapFitParams},
    'ssh_params': {'class': None},
    'uc_opt_strat_params': {'class': OptStrategyParams},
    'exp_params': {'class': OptExperimentParams}
}
for config_name, config_info in configs.items():
    config_path = dirpath_exp_local / f'{config_name}.yaml'
    config_info['data'] = load_yaml(config_path, data_class=config_info['class'])

# Mapping params
ir_map_params: IRMapConfigRateInterpFrom2DXR = configs['ir_map_params']['data']
uc_map_params: UCMapFitParams = configs['uc_map_params']['data']

# SSH params
ssh_params = configs['ssh_params']['data']
ssh_par_lethe = SSHParams(**ssh_params['lethe'])
ssh_par_grid = SSHParams(**ssh_params['grid'])

# UC optimization strategy params
uc_opt_strat_par: OptStrategyParams = configs['uc_opt_strat_params']['data']

# Experiment params
exp_params: OptExperimentParams = configs['exp_params']['data']

# Load target firing rates
fpath_target_rates = dirpath_exp_local_base / f'{target_name}.csv'
df_target_rates = pd.read_csv(fpath_target_rates)
target_rates_ = dict(zip(df_target_rates['pop_name'],
                         df_target_rates['target_rate']))
target_rates = {pop: target_rates_[pop]
                for pop in ir_map_params.pop_names}
exp_params.rr_base = target_rates


#### Folders

# Local folder to store intermediate optimization plots
dirpath_figs_local = dirpath_exp_local / 'opt_figs'
os.makedirs(dirpath_figs_local, exist_ok=True)
dirpath_figs_local_ir = dirpath_exp_local / 'opt_figs_ir'
os.makedirs(dirpath_figs_local_ir, exist_ok=True)

# Local folder to store optimization info (Ru and Rc)
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


#### I-R mapper

# Initialize I-R mapper
ir_mapper: NetIRMapper1DInterp
rate_mats: Dict[str, xr.DataArray]  # used for visualizing the I-R mapping
ir_mapper, rate_mats = ir_map_params.init_ir_mapper()

# Visualize I-R mapping
if need_plot_ir:
    dirpath_ir_plots = dirpath_exp_local / 'ir_plots'
    os.makedirs(dirpath_ir_plots, exist_ok=True)
    """ plot_ir_mapping_1d_slice(
        ir_mapper,
        rate_mats,
        ir_map_params.inp_limits,
        dirpath_ir_plots,
        target_rates
    ) """


#### U-C mapper

# Initialize UC mapping optimizer
uc_optimizer = UCOptimizer(
    uc_map_params, exp_params.pop_names,
    exp_params.rr_base, exp_params.pfr_vec,
    n_iter, ir_mapper, uc_opt_strat_par
)


#### Main part

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
        conda_env=exp_params.conda_env,
        job_cmdline_args=[exp_name_base_hpc],
        child_job_name=('j' + exp_name_base_hpc.split('/')[-1])[:10]
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
    
    # Prepare UC optimizer for the first iteration
    uc_optimizer.begin()

    # Iterations of the main optimization algorithm
    iter_num = 1
    while iter_num < n_iter:
        print(f'==== Iter: {iter_num} ====')
        
        # Rc0 -> Ru
        Ru_lst = uc_optimizer.suggest_Ru()

        # Labels of the simulations that will be added to the current batch
        sim_labels = []
        
        # Loop over the target regimes (pfr)
        for n, Ru in enumerate(Ru_lst):
            point_num = n
            print(f'Point: {point_num}')

            # Plot 2-d IR maps with selected Iu
            dirpath_figs_ir_ = dirpath_figs_local_ir / f'iter_{iter_num}_pfr_{n}'
            os.makedirs(dirpath_figs_ir_, exist_ok=True)
            if need_plot_iter_ir:        
                plot_ir_ougrid_interp(ir_mapper, Ru, dirpath_figs_ir_)
            
            # Generate a unique simulation label
            sim_label = f'req_{iter_num}_{point_num}'
            sim_labels.append(sim_label)
            
            # Check if the simulation result already exists
            if sim_res_locator.result_exists(sim_label):
                print('Simulation result already exists, do not re-run')
                continue
            
            # TODO: delete old result
            
            # Calculate an input Iu that provides the unconnected regime Ru
            Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_request = {
                'input': Iu.to_values_dict(),
                'connected': exp_params.model_cfg['connected'],
                'wmult': exp_params.model_cfg['wmult'],
                'subnet_params': exp_params.model_cfg['subnet_params']
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
        Rc_lst = NetRegime1DList()
        results_ok = True
        for n, sim_label in enumerate(sim_labels):
            print('.', end='', flush=True)
            if sim_res_locator.result_exists(sim_label):
                sim_result_desc = sim_res_locator.locate_result(sim_label)            
                Rc = get_sim_rates(
                    dk, sim_res_locator, sim_label,
                    uc_map_params.spikes_calc_params,
                    uc_map_params.rates_calc_params
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

        # Store the simulation result to UC optimizer
        uc_optimizer.store_sim_result(Ru_lst, Rc_lst)
        
        # Fit UC mapping
        try:
            uc_optimizer.fit_uc_mapper()
            uc_fit_ok = True
        except Exception as e:
            print(e)
            traceback.print_exc()
            uc_fit_ok = False

        # Save info about the current state of the optimization process
        cc = {'iter': slice(0, iter_num + 1)}
        info = {
            'pop_names': exp_params.pop_names,
            'Ru': uc_optimizer.sim_data['Ru'].isel(**cc),
            'Rc': uc_optimizer.sim_data['Rc'].isel(**cc),
            'Ru_mat_mixed': uc_optimizer.step_data['Ru'].isel(**cc),
            'Rc_mat_mixed': uc_optimizer.step_data['Rc'].isel(**cc),
            'pfr': exp_params.pfr_vec,
            'uc_mapper': uc_optimizer.uc_mappers[iter_num]
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

                plot_opt_iteration_pop(
                    pop_names=exp_params.pop_names,
                    pop_name_vis=pop_name,
                    uc_mapper=uc_optimizer.uc_mappers[iter_num],
                    Ru_lst=NetRegime1DList.from_xr(
                        uc_optimizer.step_data['Ru'].sel(iter=iter_num)), 
                    Rc_lst=NetRegime1DList.from_xr(
                        uc_optimizer.step_data['Rc'].sel(iter=iter_num)),
                    Ru_prev_lst=NetRegime1DList.from_xr(
                        uc_optimizer.step_data['Ru'].sel(iter=(iter_num - 1))),
                    Rc_prev_lst=NetRegime1DList.from_xr(
                        uc_optimizer.step_data['Rc'].sel(iter=(iter_num - 1))),
                    Rc0_lst = NetRegime1DList.from_xr(uc_optimizer.Rc0)
                )
                
                plt.get_current_fig_manager().window.showMaximized()
                plt.draw()
                plt.show()
                plt.savefig(dirpath_figs_iter / f'{pop_name}.png')

        if not uc_fit_ok:
            print('U-C map fitting failed - stop')
            break

        # Proceed to the next iteration
        uc_optimizer.next_iter()
        iter_num += 1
