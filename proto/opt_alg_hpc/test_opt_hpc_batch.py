import logging
import os
from pathlib import Path
import pickle
from pprint import pprint
import sys
import time

#sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions
import matplotlib.pyplot as plt
import numpy as np

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
from model_tuner.opt.uc_mappers import NetUCMapper1D

from model_tuner.sim_manager import SimStatus
from model_tuner.sim_manager import SimManagerHPCBatch, SimBatchPaths
from model_tuner.ssh import SSHParams, SSHClient

from filesys import FileSystem, FileSystemLocal
from proc_params import ProcStepParams, NetSpikesParams, NetRatesParams

from batch_metric_getter import BatchMetricGetter1D

from sim_result import SimResultFile
from sim_result_locator import SimResultLocator
from data_keeper import DataKeeper
from netpyne_result_parser import SimResultParserNetPyNE
from sim_data_proc import DataProcessor


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


def init_ir_mapper() -> NetIREmpiricalMapper1D:

    dirpath_batch = (
        r'D:\WORK\Salvador\repo\model_tuner\models\L24\exp_results\rx_batch_unconn_2'
    )
    exp_name = 'rx_batch_unconn_2'
    pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
    batch_param_name = 'ext_inp_rate_common'

    proc_params = {
        'net_spikes': NetSpikesParams(pop_names=pop_names),
        'net_rates': NetRatesParams(time_limits=(0.5, None))
    }

    inp_max = 1000

    # Object that exctracts firing rates from batch sim results
    bmg = BatchMetricGetter1D(
        dirpath_batch, exp_name, pop_names, batch_param_name, proc_params
    )

    # Batch parameter values that characterize the model input
    inp_rates = bmg.get_batch_par_values(batch_param_name)

    pop_rates = {}

    # Network input-to-regime mapper
    net_ir_mapper = NetIREmpiricalMapper1D()

    for pop_name in bmg.get_pop_names():
        # Request firing rates of a pop (for every batch parameter value)
        pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
        
        # Fit input-to-regime mapping for a pop
        pop_ir_mapper = PopIREmpiricalMapper1D(
            #map_type='exp_1d',
            map_type='sigmoid_1d',
            map_params = {
                'x_positive': False,
                'y_positive': True
            }
        )
        mask = (inp_rates <= inp_max)
        pop_ir_mapper.fit_from_data(inp_rates[mask], pop_rates[pop_name][mask])
        
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)
    
    return net_ir_mapper


def get_sim_regime(
        dk: DataKeeper,
        sim_res_locator: SimResultLocator,
        sim_label: str
        ) -> NetRegime1D:

    # Locate sim result by sim label
    sim_result_desc = sim_res_locator.locate_result(sim_label)
    
    # Initialize sim result parser
    res_parser = SimResultParserNetPyNE(dk=dk, result_desc=sim_result_desc)
    
    # Initialize data processor
    data_proc = DataProcessor(dk)
    
    proc_params = {
        'net_spikes': NetSpikesParams(pop_names=pop_names),
        'net_rates': NetRatesParams(time_limits=(0.5, None))
    }
    
    # Extract spikes from the sim result and strore them into dk
    spikes_data_id = res_parser.extract_net_spikes(
        proc_params['net_spikes'],
        data_name_out=f'net_spikes_{sim_label}'
    )
    
    # Load spikes from dk, calculate rates from them, save rates into dk
    rates_data_id = data_proc.calc_net_rates(
        spikes_data_id,
        proc_params['net_rates'],
        data_name_out=f'net_rates_{sim_label}'
    )
    
    # Load rates from dk
    rates = data_proc.load_data(rates_data_id)
    return NetRegime1D.from_dict(rates.data)

    
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


# Local folder for the results
dirpath_res_local = Path(
    r'D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc\data\test_opt_hpc_batch'
)

# HPC base folder
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_opt_L24_batch'

# HPC paths
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

# Batchtools script to run
fpath_batch_script_hpc = (
    '/ddn/niknovikov19/repo/model_tuner/models/L24/opt_batch_script.py'
)

# Initialize DataKeeper
dirpath_dk = str(dirpath_res_local / 'data_keeper')
os.makedirs(dirpath_dk, exist_ok=True)
dk = DataKeeper(dirpath_dk)


pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
npops = len(pop_names)

# Original target regime (pop. firing rates)
rr_base = {
    'L2e': 2.,
    'L2i': 10.,
    'L4e': 5.,
    'L4i': 15.
}

# Multipliers for the target regime
pfr_vec = np.linspace(0.1, 1.5, 5)

# Target regimes (base * pfr for each pfr)
R0_lst = []
for pfr in pfr_vec:
    rr = [rr_base[pop_name] * pfr for pop_name in pop_names]
    R0_lst.append(NetRegime1D.from_values(pop_names, rr))
R0_lst = NetRegime1DList(R0_lst)

logging.basicConfig(level=logging.ERROR, force=True)

# I-R mapper, fit a pre-calculated batch sim result
ir_mapper = init_ir_mapper()

# =============================================================================
# #Ru = NetRegime1D.from_dict(rr_base)
# Ru = R0_lst.net_regimes[-1]
# Iu = ir_mapper.R_to_I(Ru)
# a, b, c, k = ir_mapper.pop_IR_mappers['L4i']._map_func.par.values()
# x = np.linspace(-2, 1000, 200)
# y = c + a / (1 + np.exp(-k * (x - b)))
# plt.figure()
# plt.plot(x, y)
# =============================================================================

# Unconnected-to-connected regime mapper
uc_mapper = NetUCMapper1D(
    pop_names=pop_names,
    #map_type='exp_1d',
    map_type='sigmoid_1d',
    map_params = {
        'x_positive': True,
        'y_positive': True
    }
)
uc_mapper.set_to_identity()


need_delete_prev_results = 0

need_plot_iter = 1
need_plot_res = 1

n_iter = 5


with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    
    # Object that maps sim labels to sim result files
    sim_res_locator = SimResultLocator(hpc_paths.results_dir, ssh.fs)
    
    # Simulation manager
    sim_manager = SimManagerHPCBatch(
        ssh=ssh,
        fpath_batch_script=fpath_batch_script_hpc,
        batch_paths=hpc_paths,
        conda_env='netpyne_batch'
    )
    
    # Create HPC folders
    print('Create folders...')
    perm = Permissions(mode=0o777)  # Full permissions (rwxrwxrwx)
    for dirpath in hpc_paths.get_used_folders():
        ssh.fs.makedirs(dirpath, permissions=perm, recreate=True)
    
    # Delete remote files: scripts, log, results
    print('Delete old files...')
    #paths_todel = (hpc_paths.get_all_files()
    #                + [info['fpath_hpc'] for info in scripts_info.values()])
    paths_todel = []
    if need_delete_prev_results:
        paths_todel += ssh.fs.listdir(hpc_paths.results_dir)
    for path in paths_todel:
        fs_delete(ssh.fs, path)
    
    # Loop over iterations of the main optimization algorithm
    for iter_num in range(n_iter):    
        print(f'Iter: {iter_num}')
        
        # List of target regimes: scaled versions of the base target regime
        Rc_lst = R0_lst.copy()
        Rc_prev_lst = Rc_lst.copy()
        
        # Calculate unconnected regimes (Ru) from the connected target regimes (Rc)
        # using the current estimation of Rc->Ru mapping
        Ru_lst = uc_mapper.Rc_to_Ru(Rc_lst)
        
        # Labels of the simulations that will be added to the current batch
        sim_labels = []
        
        # Loop over the target regimes
        # (more precisely, over the corresponding unconnected regimes)
        for n, Ru in enumerate(Ru_lst):
            print(f'Point: {n}')
            
            # Generate a unique simulation label
            sim_label = f'req_{iter_num}_{n}'
            sim_labels.append(sim_label)
            
            if sim_res_locator.result_exists(sim_label):
                print('Simulation result already exists, do not re-run')
                continue
            
            # TODO: delete old result
            
            # Calculate an input Iu that provides the unconnected regime Ru
            Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_request = {
                'input': Iu.to_values_dict()
            }            
            sim_manager.add_sim_request(sim_label, sim_request)
            print(f'INPUT: {sim_request["input"]}')
        
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
        
        # Read the results
        print('Retrieving the results', end='', flush=True)
        for n, sim_label in enumerate(sim_labels):
            print('.', end='', flush=True)            
            sim_result_desc = sim_res_locator.locate_result(sim_label)            
            Rc_lst[n] = get_sim_regime(dk, sim_res_locator, sim_label)
        print('\nCompleted')
                
        # Re-estimate the Ru->Rc mapping based on the simulations' results
        uc_mapper.fit_from_data(Ru_lst, Rc_lst)
        
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            plt.figure()
            plt.clf()
            
            ru_mat = Ru_lst.get_pop_attr_mat('value')
            rc_mat = Rc_lst.get_pop_attr_mat('value')
            rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('value')
            iu_mat = np.full_like(ru_mat, np.nan)
            
            for m in range(ru_mat.shape[1]):
                Ru_ = NetRegime1D.from_values(pop_names, ru_mat[:, m])
                iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_inputs_vec()
                
            for n, pop in enumerate(pop_names):
                rr_u = ru_mat[n, :]
                rr_c = rc_mat[n, :]
                rr_c_prev = rc_prev_mat[n, :]
                ii_u = iu_mat[n, :]

                plt.subplot(2, npops, n + 1)
                plt.plot(ii_u, rr_u, '.')
                #ii_u_ = np.linspace(np.nanmin(ii_u), np.nanmax(ii_u), 200)
                ii_u_ = np.linspace(0, 1000, 200)
                plt.xlabel('Iu')
                plt.ylabel('Ru')
                rvis_max = rr_base[pop] * pfr_vec.max() * 1.2
                #plt.xlim(-3.5, 0)
                #plt.ylim(0, rvis_max)
                plt.title(f'pop = {pop}')
                
                plt.subplot(2, npops, npops + n + 1)
                plt.plot(rr_u, rr_c, '.')
                #rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
                rr_u_ = np.linspace(0, 100, 200)
                plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
                plt.plot(rr_u, rr_c_prev, 'kx')
                plt.xlabel('Ru')
                plt.ylabel('Rc')
                #plt.xlim(0, rvis_max)
                #plt.ylim(0, rvis_max)
            
            plt.draw()
            break
            #if need_plot_iter:
            #    if not plt.waitforbuttonpress():
            #        break