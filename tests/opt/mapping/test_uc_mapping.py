from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Qt5Agg')
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
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main'
    r'\test_opt_A1_hpc_batch_qsub\experiments'
    r'\test_1_pfr=(0.4_1.2_5)_unconn_alpha=0.25'
)

# Load UC params
fpath_uc_params = dirpath_base / 'uc_map_params.yaml'
uc_map_params: UCMapFitParams = load_yaml(fpath_uc_params,
                                          data_class=UCMapFitParams)

# Load target firing rates
df_target_rates = pd.read_csv(dirpath_base / 'target_rates.csv')
target_rates = dict(zip(df_target_rates['pop_name'],
                        df_target_rates['target_rate']))

# Load Ru and Rc matrices
iter_data = {}
for n in range(2):
    fpath_iter_data = dirpath_base / 'info' / f'Ru_Rc_req_{n}_4.pkl'
    with open(fpath_iter_data, 'rb') as f:
        iter_data[n] = pickle.load(f)

# Ru and Rc regime lists from matrices
pop_names = uc_map_params.pop_names
Ru_lst, Rc_lst = {}, {}
for n in range(2):
    Ru_lst[n] = NetRegime1DList.from_regimes_mat(pop_names, iter_data[n]['Ru'])
    Rc_lst[n] = NetRegime1DList.from_regimes_mat(pop_names, iter_data[n]['Rc'])

# Change U-C mapping parameters
uc_map_params.map_fit_params.verbose = 0
uc_map_params.map_fit_params.ftol = 0.001

#pop_name_vis = 'SOM2'
#pop_name_vis = 'PV2'
#pop_name_vis = 'NGF5A'
pop_name_vis = 'VIP2'

plt.figure()
plt.ion()

pop_id_vis = pop_names.index(pop_name_vis)
vmin = iter_data[n]['Ru'][pop_id_vis, :].min() * 0.8
vmax = iter_data[n]['Ru'][pop_id_vis, :].max() * 1.2
plt.plot([vmin, vmax], [vmin, vmax], 'k:')

rc_0 = target_rates[pop_name_vis]
plt.plot([vmin, vmax], [rc_0, rc_0], 'k--')

for k in range(2):
    # Fit U-C mapping
    uc_mapper = init_uc_mapper(uc_map_params)
    uc_fit_res = uc_mapper.fit_from_data(Ru_lst[k], Rc_lst[k],
                                        uc_map_params.map_fit_params)
    # Plot U-C mapping
    plot_opt_iteration_pop(
        pop_names, pop_name_vis, uc_mapper,
        Ru_lst[k], Rc_lst[k]
    )

plt.xlim(vmin, vmax)
plt.ylim(vmin, vmax)

plt.show()
plt.draw()

input('Press any key to continue...')
