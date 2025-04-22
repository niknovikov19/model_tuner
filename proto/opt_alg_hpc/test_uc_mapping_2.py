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
    r'\test_1_pfr=(0.4_1.0_4)_wmult=0.005_alpha=0.1'
)
dirpath_info = dirpath_base / 'info'

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

step_num = n_iter - 1

# Extract data
Ru_prev_mat = iter_data[step_num - 1]['Ru']
Rc_prev_mat = iter_data[step_num - 1]['Rc']
Ru_mat = iter_data[step_num]['Ru']
Rc_mat = iter_data[step_num]['Rc']

pop_names = iter_data[-1]['pop_names']

Ru_prev_lst = NetRegime1DList.from_regimes_mat(pop_names, Ru_prev_mat)
Rc_prev_lst = NetRegime1DList.from_regimes_mat(pop_names, Rc_prev_mat)
Ru_lst = NetRegime1DList.from_regimes_mat(pop_names, Ru_mat)
Rc_lst = NetRegime1DList.from_regimes_mat(pop_names, Rc_mat)

#uc_map_params.map_fit_params.verbose = 1
uc_map_params.map_fit_params.ftol = 0.01

# Fit uc mapper
try:
    uc_mapper = init_uc_mapper(uc_map_params)
    uc_mapper.fit_from_data(Ru_lst, Rc_lst,
                            uc_map_params.map_fit_params)
except Exception as e:
    print(f"Error fitting uc mapper: {e}")
    uc_mapper = None

# Map Rc to Ru
Ru_lst_hat = uc_mapper.Rc_to_Ru(Rc_lst)
Ru_mat_hat = Ru_lst_hat.get_pop_attr_mat('value')

pop_name_vis = 'CT5A'

plt.ion()
plt.figure()

plot_opt_iteration_pop(
    pop_names, pop_name_vis, uc_mapper,
    Ru_lst, Rc_lst, Rc_prev_lst
)

plt.show()

input("Press any key to continue...")
