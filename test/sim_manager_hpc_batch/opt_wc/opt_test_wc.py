from pathlib import Path
import pickle
from pprint import pprint
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fs.permissions import Permissions
import matplotlib.pyplot as plt
import numpy as np

from defs_wc import NetInputWC, NetRegimeWC, NetRegimeListWC
from mappers_wc import NetIRMapperWC, NetUCMapperWC
from model_wc import PopParamsWC, ModelDescWC, wc_gain, run_wc_model
from ssh_client import SSHClient
from sim_manager import SimStatus
from sim_manager_hpc_batch import SimManagerHPCBatch, SimBatchPaths
from ssh_params import SSHParams


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
dirpath_hpc_base = '/ddn/niknovikov19/test/model_tuner/test_opt_wc_batch'

# HPC paths
hpc_paths = SimBatchPaths.create_default(dirpath_base=dirpath_hpc_base)

# Scripts to run
scripts_info = {
    'batch': {'name': 'hpc_batch_script.py'},
    'job': {'name': 'hpc_job_script.py'}
}
for info in scripts_info.values():
    info['fpath_local'] = joinpath_local(dirpath_local, info['name'])
    info['fpath_hpc'] = joinpath_hpc(dirpath_hpc_base, info['name'])


def create_test_model_1pop():
    model = ModelDescWC.create_unconn(num_pops=1)
    model.conn[0, 0] = -0.1
    return model

def create_test_model_2pop():
    model = ModelDescWC.create_unconn(num_pops=2)
    model.conn = np.array([[0.1, -0.2], [0.2, -0.1]])
    return model

# Network of Wilson-Cowan populations
#model = create_test_model_1pop()
model = create_test_model_2pop()

pop_names = model.get_pop_names()
npops = len(pop_names)

# Original target regime (vector of pop. firing rates)
rr_base = np.arange(npops) + 1

# P_FR
pfr_vec = np.linspace(0.1, 1.5, 10)

# Target regimes (base * pfr for each pfr)
R0_lst = NetRegimeListWC(
    [NetRegimeWC.from_rates(pop_names, rr_base * pfr) for pfr in pfr_vec]
)

# I-R mapper, explicitly uses Wilson-Cowan gain functions of populations
ir_mapper = NetIRMapperWC(model)

# Unconnected-to-connected regime mapper
#uc_mapper = NetUCMapperWC(pop_names, 'exp')
uc_mapper = NetUCMapperWC(pop_names, 'sigmoid')
uc_mapper.set_to_identity()

# Params of WC model simulations
sim_par = {'niter': 20, 'dr_mult': 1}

need_delete_prev_results = 0

need_plot_iter = 1
need_plot_res = 1

n_iter = 5

with SSHClient(
        ssh_par_fs=ssh_par_lethe,
        ssh_par_conn=[ssh_par_lethe, ssh_par_grid]
        ) as ssh:
    
    # Simulation manager
    sim_manager = SimManagerHPCBatch(
        ssh=ssh,
        fpath_batch_script=scripts_info['batch']['fpath_hpc'],
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
    paths_todel = (hpc_paths.get_all_files()
                    + [info['fpath_hpc'] for info in scripts_info.values()])
    if need_delete_prev_results:
        paths_todel += ssh.fs.listdir(hpc_paths.results_dir)
    for path in paths_todel:
        fs_delete(ssh.fs, path)
        
    # Upload the scripts (batch and job) to HPC
    print('Upload batch and job scripts...')
    for info in scripts_info.values():
        ssh.fs.upload_file(info['fpath_local'], info['fpath_hpc'])
    
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
            
            # Calculate an input Iu that provides the unconnected regime Ru
            Iu = ir_mapper.R_to_I(Ru)
            
            # Add a request for simulation with the input Iu (non-blocking)
            sim_label = f'req_{iter_num}_{n}'
            sim_request = {
                'wc_model': model,
                'wc_sim_params': sim_par,
                'Iext': Iu,
                'R0': Ru
            }            
            sim_manager.add_sim_request(sim_label, sim_request)
            sim_labels.append(sim_label)
        
        # Push simulation requests
        print('Push simulation requests to HPC', flush=True)
        sim_manager.push_all_requests()
        
        # Wait for simulation results
        print('Waiting for completion', end='', flush=True)
        while not sim_manager.is_finished():
            print('.', end='', flush=True)
            time.sleep(0.5)
        print()
        pprint(sim_manager.get_all_sim_statuses())
        
        # Read the results
        print('Retrieving the results', end='', flush=True)
        for n, label in enumerate(sim_labels):
            print('.', end='', flush=True)
            if sim_manager.get_sim_status(label) == SimStatus.DONE:
                fpath_res = sim_manager.get_sim_result_path(label)
                with ssh.fs.open(fpath_res, 'rb') as fid:
                    sim_result = pickle.load(fid)
                Rc_lst[n] = sim_result['R']
            else:
                raise RuntimeError(f'Simulation {label} has ERROR status')
        print()
                
        # Re-estimate the Ru->Rc mapping based on the simulations' results
        uc_mapper.fit_from_data(Ru_lst, Rc_lst)
        
        if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
            plt.figure(115)
            plt.clf()
            
            ru_mat = Ru_lst.get_pop_attr_mat('r')
            rc_mat = Rc_lst.get_pop_attr_mat('r')
            rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('r')
            iu_mat = np.full_like(ru_mat, np.nan)
            
            for m in range(ru_mat.shape[1]):
                Ru_ = NetRegimeWC.from_rates(pop_names, ru_mat[:, m])
                iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_attr_vec('I')
                
            for n, pop in enumerate(pop_names):
                rr_u = ru_mat[n, :]
                rr_c = rc_mat[n, :]
                rr_c_prev = rc_prev_mat[n, :]
                ii_u = iu_mat[n, :]

                plt.subplot(2, npops, n + 1)
                plt.plot(ii_u, rr_u, '.')
                ii_u_ = np.linspace(np.nanmin(ii_u), np.nanmax(ii_u), 200)
                plt.plot(ii_u_, wc_gain(ii_u_, model.pops[pop]))
                plt.xlabel('Iu')
                plt.ylabel('Ru')
                rvis_max = rr_base[n] * pfr_vec.max() * 1.2
                plt.xlim(-3.5, 0)
                plt.ylim(0, rvis_max)
                plt.title(f'pop = {pop}')
                
                plt.subplot(2, npops, npops + n + 1)
                plt.plot(rr_u, rr_c, '.')
                rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
                plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
                plt.plot(rr_u, rr_c_prev, 'kx')
                plt.xlabel('Ru')
                plt.ylabel('Rc')
                plt.xlim(0, rvis_max)
                plt.ylim(0, rvis_max)
            
            plt.draw()
            #if need_plot_iter:
            #    if not plt.waitforbuttonpress():
            #        break