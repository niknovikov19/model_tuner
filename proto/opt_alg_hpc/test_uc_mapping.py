from pathlib import Path
import pickle

import numpy as np

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.uc_mappers import NetUCMapper1D

from model_tuner.main import (
    UCMapFitParams,
    init_uc_mapper
)

from model_tuner.utils import load_yaml


dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_opt_L24_hpc_batch'
    r'\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.25_alpha=0.25'
)
dirpath_old = dirpath_base / 'old' / 'info'
dirpath_new = dirpath_base / 'info'

pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
npops = len(pop_names)

data = {'old': {}, 'new': {}}

# Load data
for n in range(2):    
    fname = f'Ru_Rc_req_{n}_6.pkl'
    with open(dirpath_old / fname, 'rb') as fid:
        data['old'][n] = pickle.load(fid)
    with open(dirpath_new / fname, 'rb') as fid:
        data['new'][n] = pickle.load(fid)

# Load uc map params
uc_map_params: UCMapFitParams = load_yaml(
    dirpath_base / 'uc_map_params.yaml',
    data_class=UCMapFitParams
)

for data_type in ['old', 'new']:

    # Extract data
    Ru0_mat = data[data_type][0]['Ru']
    Ru1_mat = data[data_type][1]['Ru']
    Rc0_mat = data[data_type][0]['Rc']
    
    Ru_lst = NetRegime1DList.from_regimes_mat(pop_names, Ru0_mat)
    Rc_lst = NetRegime1DList.from_regimes_mat(pop_names, Rc0_mat)

    # Fit uc mapper
    uc_mapper = init_uc_mapper(uc_map_params)
    uc_mapper.fit_from_data(Ru_lst, Rc_lst)

    # Map Rc to Ru
    Ru1_lst_hat = uc_mapper.Rc_to_Ru(Rc_lst)
    Ru1_mat_hat = Ru1_lst_hat.get_pop_attr_mat('value')

    # Print
    pop = 0
    print(f'\n==== {data_type} ====')
    print('Ru0:')
    print(np.round(Ru0_mat[pop, :], 3))
    print('Rc0:')
    print(np.round(Rc0_mat[pop, :], 3))
    print('Ru1:')
    print(np.round(Ru1_mat[pop, :], 3))
    print('Ru1_hat:')
    print(np.round(Ru1_mat_hat[pop, :], 3))
