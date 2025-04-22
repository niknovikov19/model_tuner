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

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.uc_mappers import NetUCMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D

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
    IRMapConfigRateFrom1DSim,
    UCMapFitParams,
    OptExperimentParams,
    init_uc_mapper,
    get_sim_rates,
    plot_opt_iteration
)

from model_tuner.utils import load_yaml, compare_yaml, yaml_diff

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


# Local base folder (configs in the root, results in subfolders)
dirpath_base_local = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_opt_L24_hpc_batch_qsub'
)

# Load config files
configs = {
    'ir_map_params': {'class': IRMapConfigRateFrom1DSim},
    'uc_map_params': {'class': UCMapFitParams},
    'ssh_params': {'class': None}
}
for config_name, config_info in configs.items():
    config_path = dirpath_base_local / f'{config_name}.yaml'
    config_info['data'] = load_yaml(config_path, data_class=config_info['class'])
ir_map_params: IRMapConfigRateFrom1DSim = configs['ir_map_params']['data']
uc_map_params: UCMapFitParams = configs['uc_map_params']['data']
ssh_params = configs['ssh_params']['data']

ssh_par_lethe = SSHParams(**ssh_params['lethe'])
ssh_par_grid = SSHParams(**ssh_params['grid'])

# Parameters of the model tuning experiment
exp_params = OptExperimentParams(
    fpath_batch_script_hpc = (
        '/ddn/niknovikov19/repo/model_tuner/models/L24/opt_batch_script.py'
    ),
    dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_opt_L24_batch_qsub',
    conda_env='netpyne_batch',
    pop_names=('L2e', 'L2i', 'L4e', 'L4i'),
    rr_base={
        'L2e': 2.,
        'L2i': 10.,
        'L4e': 5.,
        'L4i': 15.
    },
    pfr_vec=np.linspace(0.1, 1.5, 7),
    uc_alpha=0.25,
    wmult=0.25
)

config_info['exp_params'] = {
    'class': OptExperimentParams,
    'data': exp_params
}

# Number of iterations
# (don't put it to config, so it can be increased later)
n_iter = 15

def _gen_exp_name(exp_params: OptExperimentParams) -> str:
    rr_str = 'exp_r0=({})'.format(
        '_'.join([str(int(r)) for r in exp_params.rr_base.values()])
    )
    pfr_str = 'pfr=({}_{}_{})'.format(
        exp_params.pfr_vec.min(),
        exp_params.pfr_vec.max(),
        len(exp_params.pfr_vec)
    )
    param_str = 'wmult={}_alpha={}'.format(
        exp_params.wmult,
        exp_params.uc_alpha
    )
    return f'{rr_str}_{pfr_str}_{param_str}'

exp_name = _gen_exp_name(exp_params)
#print(exp_name)

# Local folder to store experiment results
dirpath_res_local = dirpath_base_local / 'experiments' / exp_name
os.makedirs(dirpath_res_local, exist_ok=True)

# Copy config files to the experiment folder if needed
for config_name, config_info in configs.items():
    fpath_cfg_exp = dirpath_res_local / f'{config_name}.yaml'
    if fpath_cfg_exp.exists():
        # If a config file already exists in the experiment folder,
        # it should match the original config from the base folder
        cfg_prev = load_yaml(fpath_cfg_exp, data_class=config_info['class'])
        if not compare_yaml(config_info['data'], cfg_prev):
            err_str = (
                f'Config file {config_name}.yaml from the experiment folder'
                'does not match the original file from the base folder.'
            )
            print(err_str)
            pprint(yaml_diff(config_info['data'], cfg_prev))
            raise Exception(err_str)
    else:
        # Copy the original config file to the experiment folder
        fpath_cfg_base = dirpath_base_local / f'{config_name}.yaml'
        shutil.copy(fpath_cfg_base, fpath_cfg_exp)

# Local folder to store intermediate optimization plots
dirpath_figs_local = dirpath_res_local / 'opt_figs'
os.makedirs(dirpath_figs_local, exist_ok=True)
# Local folder to store optimization info (Ru and Rc)
dirpath_info = dirpath_res_local / 'info'
os.makedirs(dirpath_info, exist_ok=True)

# HPC base folder for the experiment data
exp_name_hpc = exp_name.replace('=', '_').replace('(', '').replace(')', '')
dirpath_hpc_base = exp_params.dirpath_hpc_base + '/' + exp_name_hpc
# HPC paths used in batch simulations
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

# Initialize DataKeeper
dirpath_dk = str(dirpath_res_local / 'data_keeper')
os.makedirs(dirpath_dk, exist_ok=True)
dk = DataKeeper(dirpath_dk)

logging.basicConfig(level=logging.ERROR, force=True)

# Initialize input-to-regime mapper: fit a pre-calculated batch sim result
ir_mapper: NetIREmpiricalMapper1D = (
    ir_map_params.init_ir_mapper(need_plot=True)
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

need_delete_prev_results = 0
need_plot_iter = 1
need_plot_res = 1

with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    
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
                'wmult': exp_params.wmult
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
        info = {'Ru': Ru_mat, 'Rc': Rc_mat}
        fpath_info = dirpath_info / f'Ru_Rc_{sim_label}.pkl'
        with open(fpath_info, 'wb') as fid:
            pickle.dump(info, fid)        
                
        # Re-estimate the Ru->Rc mapping based on the simulations' results
        uc_fit_res = uc_mapper.fit_from_data(Ru_lst, Rc_lst)
        
        # Visualize the iteration result
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            set_qt_backend()
            plt.ion()
            plt.figure(111)
            plot_opt_iteration(
                exp_params.pop_names, ir_mapper, uc_mapper,
                Ru_lst, Rc_lst, Rc_prev_lst, Rc0_lst
            )
            plt.get_current_fig_manager().window.showMaximized()
            plt.draw()
            plt.show()
            plt.savefig(dirpath_figs_local / f'opt_iter={iter_num}.png')

        if not uc_fit_res:
            print('U-C map fitting failed - stop')
            break
