from glob import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

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
    #r'\test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=0.2'
    r'\test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=1'
)
dirpath_info = dirpath_base / 'info'

pop_name_vis = 'SOM6'

step_num = -1


# Load Ru and Rc data
iter_data = []
n_iter = len(glob(str(dirpath_info / 'Ru_Rc_req_*_*.pkl')))
for n in range(n_iter):
    fpath_mask = str(dirpath_info / f'Ru_Rc_req_{n}_*.pkl')
    fpath_iter_data = glob(fpath_mask)[0]
    with open(fpath_iter_data, 'rb') as f:
        iter_data.append(pickle.load(f))

# Load uc map params
uc_map_params: UCMapFitParams = load_yaml(
    dirpath_base / 'uc_map_params.yaml',
    data_class=UCMapFitParams
)

if step_num < 0:
    step_num = n_iter + step_num

# Extract data
if step_num > 0:
    Ru_prev_mat = iter_data[step_num - 1]['Ru']
    Rc_prev_mat = iter_data[step_num - 1]['Rc']
else:
    Ru_prev_mat = iter_data[step_num]['Ru']
    Rc_prev_mat = iter_data[step_num]['Ru']
Ru_mat = iter_data[step_num]['Ru']
Rc_mat = iter_data[step_num]['Rc']

pop_names = iter_data[-1]['pop_names']

Ru_prev_lst = NetRegime1DList.from_regimes_mat(pop_names, Ru_prev_mat)
Rc_prev_lst = NetRegime1DList.from_regimes_mat(pop_names, Rc_prev_mat)
Ru_lst = NetRegime1DList.from_regimes_mat(pop_names, Ru_mat)
Rc_lst = NetRegime1DList.from_regimes_mat(pop_names, Rc_mat)

rc_min = Rc_lst[0][pop_name_vis].value
rc_max = Rc_lst[-1][pop_name_vis].value

#uc_map_params.map_fit_params.verbose = 1
#uc_map_params.map_fit_params.ftol = 0.01
uc_map_params.map_type = 'richards_1d'
uc_map_params.fit_param_bounds = {
    #'c': (0, rc_min),
    'a': (0, np.inf),
    'q': (1, 30),
    #'b': (0, np.inf)
}

uc_mapper = init_uc_mapper(uc_map_params)

# Fit uc mapper
try:
    uc_mapper.fit_from_data(
        Ru_lst, Rc_lst,
        fit_params=uc_map_params.map_fit_params,
        bounds=uc_map_params.fit_param_bounds
    )
except Exception as e:
    print(f"Error fitting uc mapper: {e}")
    uc_mapper = None

# Map Rc to Ru
Ru_lst_hat = uc_mapper.Rc_to_Ru(Rc_lst)
Ru_mat_hat = Ru_lst_hat.get_pop_attr_mat('value')

print(f'{pop_name_vis}: {uc_mapper._map_funcs[pop_name_vis].par}')

plt.ion()
plt.figure()

plot_opt_iteration_pop(
    pop_names, pop_name_vis, uc_mapper,
    Ru_lst, Rc_lst, Rc_prev_lst,
    ru_limits=(0, 15)
)

plt.show()

input("Press any key to continue...")
