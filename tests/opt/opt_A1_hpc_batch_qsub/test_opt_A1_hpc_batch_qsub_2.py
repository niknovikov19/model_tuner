import logging
import os
from pathlib import Path
import pickle
from pprint import pprint
import shutil
import sys
import time
from typing import Dict

from fs.permissions import Permissions
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import NetIRMapper1DSlice
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
    IRMapConfigRateFrom2DBatchResLists,
    UCMapFitParams,
    OptExperimentParams,
    init_uc_mapper,
    get_sim_rates,
    plot_opt_iteration_pop,
    plot_ir_mapping_1d_slice
)

from model_tuner.utils import load_yaml, save_yaml, compare_yaml, yaml_diff

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
exp_name = 'test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=1'

# Number of iterations
# (don't put it to config, so it can be increased later)
n_iter = 10

# Local experiment folder
dirpath_exp_local = dirpath_base_local / 'experiments' / exp_name

# Load config files
configs = {
    'ir_map_params': {'class': IRMapConfigRateFrom2DBatchResLists},
    'uc_map_params': {'class': UCMapFitParams},
    'ssh_params': {'class': None},
    'exp_params': {'class': OptExperimentParams}
}
for config_name, config_info in configs.items():
    config_path = dirpath_exp_local / f'{config_name}.yaml'
    config_info['data'] = load_yaml(config_path, data_class=config_info['class'])

# Mapping params
ir_map_params: IRMapConfigRateFrom2DBatchResLists = configs['ir_map_params']['data']
uc_map_params: UCMapFitParams = configs['uc_map_params']['data']

# SSH params
ssh_params = configs['ssh_params']['data']
ssh_par_lethe = SSHParams(**ssh_params['lethe'])
ssh_par_grid = SSHParams(**ssh_params['grid'])

# Experiment params
exp_params: OptExperimentParams = configs['exp_params']['data']

# Load target firing rates
fpath_target_rates = dirpath_exp_local / 'target_rates.csv'
df_target_rates = pd.read_csv(fpath_target_rates)
target_rates_ = dict(zip(df_target_rates['pop_name'],
                         df_target_rates['target_rate']))
target_rates = {pop: target_rates_[pop]
                for pop in ir_map_params.pop_names}
exp_params.rr_base = target_rates

# Local folder to store intermediate optimization plots
dirpath_figs_local = dirpath_exp_local / 'opt_figs'
os.makedirs(dirpath_figs_local, exist_ok=True)

# Local folder to store optimization info (Ru and Rc)
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

# Action flags
need_delete_prev_results = 0
need_recalc_ir = 0
need_plot_ir = 0
need_plot_iter = 1
need_plot_res = 1

# Initialize I-R mapper
ir_mapper: NetIRMapper1DSlice
rate_mats: Dict[str, xr.DataArray]  # used for visualizing the I-R mapping
fpath_ir_mapper = dirpath_exp_local / 'ir_mapper.pkl'
if os.path.exists(fpath_ir_mapper) and not need_recalc_ir:
    # Load previously fitted I-R mapping and from pkl
    with open(fpath_ir_mapper, 'rb') as fid:
        ir_mapper, rate_mats = pickle.load(fid)
else:
    # Fit I-R mapping to batch simulation results and save it to pkl
    ir_mapper, rate_mats = ir_map_params.init_ir_mapper()
    with open(fpath_ir_mapper, 'wb') as fid:
        pickle.dump((ir_mapper, rate_mats), fid)

# Visualize I-R mapping
if need_plot_ir:
    dirpath_ir_plots = dirpath_exp_local / 'ir_plots'
    os.makedirs(dirpath_ir_plots, exist_ok=True)
    plot_ir_mapping_1d_slice(
        ir_mapper,
        rate_mats,
        ir_map_params.inp_limits,
        dirpath_ir_plots,
        target_rates
    )

# Initialize unconnected-to-connected regime mapper: set to identity
uc_mapper: NetUCMapper1D = init_uc_mapper(uc_map_params)

# Target regimes (base * pfr for each pfr)
def gen_target_regimes_list(
        exp_params_: OptExperimentParams
        ) -> NetRegime1DList:
    Rc0_lst_ = []
    for pfr in exp_params_.pfr_vec:
        rr = [exp_params_.rr_base[pop_name] * pfr
              for pop_name in exp_params_.pop_names]
        Rc0_lst_.append(
            NetRegime1D.from_values(exp_params_.pop_names, rr)
        )
    return NetRegime1DList(Rc0_lst_)
Rc0_lst = gen_target_regimes_list(exp_params)

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
    
    # Loop over iterations of the main optimization algorithm
    for iter_num in range(n_iter):
        
        print(f'==== Iter: {iter_num} ====')
        
        # Calculate unconnected regimes (Ru) from the connected target regimes (Rc)
        # using the current estimation of Rc->Ru mapping
        Ru_lst_ = uc_mapper.Rc_to_Ru(Rc0_lst)
        
        # Discard points for which Rc->Ru mapping failed
        Ru_lst = NetRegime1DList()
        Rc_lst = NetRegime1DList()
        valid_points = []
        for n, (Ru, Rc) in enumerate(zip(Ru_lst_, Rc0_lst)):
            if Ru.is_valid():
                Ru_lst.append(Ru.copy())
                Rc_lst.append(Rc.copy())
                valid_points.append(n)
            else:
                print(f'Rc->Ru mapping failed for the point {n}')

        # Store a copy of Rc_lst to use it later.
        # Rc_lst itself will be updated based on simulation results.
        Rc_prev_lst = Rc_lst.copy()

        # Labels of the simulations that will be added to the current batch
        sim_labels = []
        
        # Loop over the target regimes
        # (more precisely, over the corresponding unconnected regimes)
        for n, Ru in enumerate(Ru_lst):
            
            point_num = valid_points[n]
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
            Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_request = {
                'input': Iu.to_values_dict(),
                'connected': exp_params.model_cfg['connected'],
                'wmult': exp_params.model_cfg['wmult']
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
        for n, sim_label in enumerate(sim_labels):
            print('.', end='', flush=True)            
            sim_result_desc = sim_res_locator.locate_result(sim_label)            
            Rc_lst[n] = get_sim_rates(
                dk, sim_res_locator, sim_label,
                uc_map_params.spikes_calc_params,
                uc_map_params.rates_calc_params
            )
        print('\nCompleted')
        
        # Mix old and new regimes
        Rc_lst = NetRegime1DList.mix(Rc_prev_lst, Rc_lst, exp_params.uc_alpha)
        
        # Save info about the current state of the optimization process
        Ru_mat = Ru_lst.get_pop_regimes_mat()
        Rc_mat = Rc_lst.get_pop_regimes_mat()
        info = {
            'pop_names': exp_params.pop_names,
            'Ru': Ru_mat,
            'Rc': Rc_mat,
            'pfr': exp_params.pfr_vec[valid_points],
        }
        fpath_info = dirpath_info / f'Ru_Rc_{sim_label}.pkl'
        with open(fpath_info, 'wb') as fid:
            pickle.dump(info, fid)        
                
        # Re-estimate the Ru->Rc mapping based on the simulations' results
        uc_fit_res = uc_mapper.fit_from_data(
            Ru_lst, Rc_lst,
            uc_map_params.map_fit_params
        )
        
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
                    exp_params.pop_names, pop_name, uc_mapper,
                    Ru_lst, Rc_lst, Rc_prev_lst, Rc0_lst
                )
                
                plt.get_current_fig_manager().window.showMaximized()
                plt.draw()
                plt.show()
                plt.savefig(dirpath_figs_iter / f'{pop_name}.png')

        if not uc_fit_res:
            print('U-C map fitting failed - stop')
            break
