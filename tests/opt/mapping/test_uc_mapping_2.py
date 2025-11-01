from glob import glob
from pathlib import Path
import pickle

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.uc_mappers import NetUCMapper1D

from model_tuner.main import (
    UCMapFitParams,
    init_uc_mapper,
    plot_opt_iteration_pop
)

from model_tuner.utils import load_yaml


dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\opt_experiments\ou_surr_to_full'
    r'\data\exp_ou_full_state1_mech1_nosub_wmult_0.1'
)

exp_name = 'all_conn_5s_spline_alpha_0.5_auto_0.5_pfr_2'

# Load UC params
fpath_uc_params = dirpath_base / exp_name / 'uc_map_params.yaml'
uc_map_params: UCMapFitParams = load_yaml(fpath_uc_params,
                                          data_class=UCMapFitParams)

# Load target firing rates
df_target_rates = pd.read_csv(dirpath_base / 'target_state_1.csv')
target_rates = dict(zip(df_target_rates['pop_name'],
                        df_target_rates['target_rate']))

fpath_mask = str(dirpath_base / exp_name / 'info' / 'Ru_Rc_req_*.pkl')
n_iter = len(glob(fpath_mask))

# Load Ru and Rc matrices
iter_data = {}
for n in range(1, n_iter + 1):
    fpath_iter_data = dirpath_base / exp_name / 'info' / f'Ru_Rc_req_{n}_3.pkl'
    with open(fpath_iter_data, 'rb') as f:
        iter_data[n] = pickle.load(f)

# Ru and Rc regime lists from matrices
pop_names = uc_map_params.pop_names
Ru_lst, Rc_lst = {}, {}
for n in range(1, n_iter + 1):
    Ru_lst[n] = NetRegime1DList.from_regimes_mat(
        pop_names, iter_data[n]['Ru_mat_mixed'].isel(iter=-1))
    Rc_lst[n] = NetRegime1DList.from_regimes_mat(
        pop_names, iter_data[n]['Rc_mat_mixed'].isel(iter=-1))

# Change U-C mapping parameters
#uc_map_params.map_type = 'spline_1d'
#uc_map_params.map_fit_params.verbose = 0
#uc_map_params.map_fit_params.ftol = 1e-4
#uc_map_params.map_fit_params.gtol = 1e-10
#uc_map_params.map_fit_params.max_nfev = 10000
#uc_map_params.map_fit_params.return_first_guess = 0
#uc_map_params.fit_param_bounds = {}
#uc_map_params.fit_param_bounds['y0'] = (0, 0.1)
#uc_map_params.fit_param_bounds['c'] = (0, 0.1)

pfr_vec = iter_data[1]['pfr']

#pop_name_vis = 'SOM4'
pop_name_vis = 'PV2'

#plt.figure()
#plt.ion()

pop_id_vis = pop_names.index(pop_name_vis)
vmin = iter_data[n]['Ru'][pop_id_vis, :].min() * 0.5
vmax = iter_data[n]['Ru'][pop_id_vis, :].max() * 2
#vmin = np.minimum(0, vmin)
#vmax = np.maximum(7, vmax)
plt.plot([vmin, vmax], [vmin, vmax], 'k:')

rc_0 = target_rates[pop_name_vis]
#rc_0 = 2

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for k in range(1, n_iter + 1):
    # Fit U-C mapping
    uc_mapper = init_uc_mapper(uc_map_params)
    uc_fit_res = uc_mapper.fit_from_data(
        Ru_lst[k], Rc_lst[k], uc_map_params.map_fit_params,
        bounds=uc_map_params.fit_param_bounds,
        verbose=0
    )
    # Plot U-C mapping
    """ plot_opt_iteration_pop(
        pop_names, pop_name_vis, uc_mapper,
        Ru_lst[k], Rc_lst[k], color=colors[k - 1],
        ru_limits=(vmin, vmax)
    ) """
    # Inverse
    for pfr in pfr_vec:
        rc_0_ = rc_0 * pfr
        plt.plot([vmin, vmax], [rc_0_, rc_0_], 'k--')
        ru_0 = uc_mapper.pop_UC_mappers[pop_name_vis].Rc_to_Ru(rc_0_).value
        plt.plot(ru_0, rc_0_, 'o', color=colors[k - 1])

#plt.xlim(4, 6)
#plt.ylim(-1, 10)

plt.show()
#plt.draw()
#input('Press any key to continue...')
