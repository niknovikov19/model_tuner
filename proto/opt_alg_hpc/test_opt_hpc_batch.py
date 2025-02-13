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
from model_tuner.opt.map_funcs import MapFuncType

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

    r_limits = {
        'L2e': (0, 250),
        'L2i': (0, 1000),
        'L4e': (0, 250),
        'L4i': (0, 1000),
    }
    
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
            map_type=MapFuncType.RICHARDS_1D,
            map_params = {
                'x_positive': True,
                'y_positive': True
            }
        )
        rlim = r_limits[pop_name]
        mask = (inp_rates >= rlim[0]) & (inp_rates <= rlim[1])
        xx = inp_rates[mask]
        yy = pop_rates[pop_name][mask]
        ww = np.clip(yy ** 0.5, 0.1, 5)
        pop_ir_mapper.fit_from_data(xx, yy, ww)
        
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)
    
    need_plot = 1
    if need_plot:
        plot_ir_mapping(bmg, batch_param_name, net_ir_mapper)
    
    return net_ir_mapper


def plot_ir_mapping(
        bmg: BatchMetricGetter1D,
        batch_param_name: str,
        net_ir_mapper: NetIREmpiricalMapper1D
        ):

    # Inputs and outputs used for fitting
    inp_rates = bmg.get_batch_par_values(batch_param_name)
    pop_rates = {}
    for pop_name in bmg.get_pop_names():
        pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
        
    r_limits = {
        'L2e': (0, 250),
        'L2i': (100, 750),
        'L4e': (0, 250),
        'L4i': (100, 750),
    }

    # Apply I-R mapping to a range of input rates
    n_points = 100
    rr_inp, rr_pop = {}, {}
    for pop_name in bmg.get_pop_names():
        rlim = r_limits[pop_name]
        rr_inp[pop_name] = np.linspace(rlim[0], rlim[1], n_points)
        rr_pop[pop_name] = np.zeros(n_points)
        for n, r_inp in enumerate(rr_inp[pop_name]):
            pop_inputs = {pop_name_: PopInput1D(value=r_inp)
                          for pop_name_ in pop_names}
            net_input = NetInput1D(pop_inputs=pop_inputs)
            net_regime = net_ir_mapper.I_to_R(net_input)        
            rr_pop[pop_name][n] = net_regime.pop_regimes[pop_name].value
    
    plt.figure()    
    for n, pop_name in enumerate(pop_names):
        plt.subplot(1, len(pop_names), n + 1)
        
        # Fitted data produced by the I-R mapper
        x, y = rr_inp[pop_name], rr_pop[pop_name]
        plt.plot(x, y)
        
        # Data used to "learn" the I-R mapping
        plt.plot(inp_rates, pop_rates[pop_name], 'k.')
        
        plt.title(pop_name)
        plt.xlabel('Input rate')
        plt.ylabel('Pop. rate')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())


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
        data_name_out=f'net_rates_{sim_label}',
        recalc=False
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
pfr_vec = np.linspace(0.1, 1.5, 7)

# Rate of U-C mapping change between iterations (0 = old, 1 = replace)
uc_alpha = 0.25

# Global weight multiplier
wmult = 0.25

# Experiment name
rr_str = 'r0=(' + '_'.join([str(int(r)) for r in rr_base.values()]) + ')'
pfr_str = f'pfr=({pfr_vec.min()}_{pfr_vec.max()}_{len(pfr_vec)})'
param_str = f'wmult={wmult}_alpha={uc_alpha}'
exp_name = f'exp_{rr_str}_{pfr_str}_{param_str}'
#print(exp_name)

need_delete_prev_results = 0

need_plot_iter = 1
need_plot_res = 1

n_iter = 50


# Local folder for the results
dirpath_res_local = Path(
    r'D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc\data\test_opt_hpc_batch'
)
dirpath_res_local = dirpath_res_local / exp_name
os.makedirs(dirpath_res_local, exist_ok=True)

# HPC base folder
exp_name_hpc = exp_name.replace('=', '_').replace('(', '').replace(')', '')
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_opt_L24_batch'
dirpath_hpc_base = dirpath_hpc_base + '/' + exp_name_hpc

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

# Folder to store intermediate optimization plots
dirpath_figs_local = dirpath_res_local / 'opt_figs'
os.makedirs(dirpath_figs_local, exist_ok=True)


# Target regimes (base * pfr for each pfr)
Rc0_lst = []
for pfr in pfr_vec:
    rr = [rr_base[pop_name] * pfr for pop_name in pop_names]
    Rc0_lst.append(NetRegime1D.from_values(pop_names, rr))
Rc0_lst = NetRegime1DList(Rc0_lst)

logging.basicConfig(level=logging.ERROR, force=True)

# I-R mapper, fit a pre-calculated batch sim result
ir_mapper = init_ir_mapper()

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


#logging.basicConfig(level=logging.DEBUG, force=True)


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
        print(f'==== Iter: {iter_num} ====')
        
        # Calculate unconnected regimes (Ru) from the connected target regimes (Rc)
        # using the current estimation of Rc->Ru mapping
        Ru_lst_ = uc_mapper.Rc_to_Ru(Rc0_lst)
        
        # Discart points for which Rc->Ru mapping failed
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
            
            if sim_res_locator.result_exists(sim_label):
                print('Simulation result already exists, do not re-run')
                continue
            
            # TODO: delete old result
            
            # Calculate an input Iu that provides the unconnected regime Ru
            Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_request = {
                'input': Iu.to_values_dict(),
                'wmult': wmult
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
            Rc_lst[n] = get_sim_regime(dk, sim_res_locator, sim_label)
        print('\nCompleted')
        
        # Mix old and new regimes
        Rc_lst = NetRegime1DList.mix(Rc_prev_lst, Rc_lst, uc_alpha)
        
        Ru_mat = Ru_lst.get_pop_regimes_mat()
        Rc_mat = Rc_lst.get_pop_regimes_mat()
        info = {'Ru': Ru_mat, 'Rc': Rc_mat}
        dirpath_info = dirpath_res_local / 'info'
        os.makedirs(dirpath_info, exist_ok=True)
        fpath_info = dirpath_info / f'Ru_Rc_{sim_label}.pkl'
        with open(fpath_info, 'wb') as fid:
            pickle.dump(info, fid)        
                
        # Re-estimate the Ru->Rc mapping based on the simulations' results
        uc_fit_res = uc_mapper.fit_from_data(Ru_lst, Rc_lst)
        
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            plt.figure(111)
            plt.clf()
            
            ru_mat = Ru_lst.get_pop_attr_mat('value')
            rc_mat = Rc_lst.get_pop_attr_mat('value')
            rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('value')
            rc0_mat = Rc0_lst.get_pop_attr_mat('value')
            iu_mat = np.full_like(ru_mat, np.nan)
            
            for m in range(ru_mat.shape[1]):
                Ru_ = NetRegime1D.from_values(pop_names, ru_mat[:, m])
                iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_inputs_vec()
                
            for n, pop in enumerate(pop_names):
                rr_u = ru_mat[n, :]
                rr_c = rc_mat[n, :]
                rr_c_prev = rc_prev_mat[n, :]
                rr_c0 = rc0_mat[n, :]
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
                rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
                #rr_u_ = np.linspace(0, 100, 200)
                plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
                plt.plot(rr_u, rr_c_prev, 'kx')
                plt.xlabel('Ru')
                plt.ylabel('Rc')
                plt.xlim(0, rr_c0.max() * 2)
                plt.ylim(0, rr_c0.max() * 1.2)
            
            plt.draw()
            plt.savefig(dirpath_figs_local / f'opt_iter={iter_num}.png')
        
        if not uc_fit_res:
            print('U-C map fitting failed - stop')
            break
        
        #break
