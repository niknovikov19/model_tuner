import os
from pathlib import Path
import pickle

import numpy as np
import xarray as xr


dirpath_in = Path(
    r"D:\WORK\Salvador\repo\model_tuner\test_data\main"
    r"\test_opt_A1_hpc_batch_qsub\experiments"
    r"\test_2_pfr=(0.4_1.0_4)_wmult=0.01_alpha=1\info"
)

# Count the iterations
n_iter = len(list(dirpath_in.rglob('Ru_Rc_req_*_3.pkl')))

# Open the first iter to get coord info
fpath_in = dirpath_in / f'Ru_Rc_req_0_3.pkl'
with open(fpath_in, 'rb') as fid:
    X_ = pickle.load(fid)
pop_names, pfr_vec = X_['pop_names'], X_['pfr']

# Allocate output arrays
sz = len(pop_names), len(pfr_vec), n_iter
Z = xr.DataArray(
    np.full(sz, np.nan),
    dims=['pop', 'pfr', 'iter'],
    coords={'pop': list(pop_names),
            'pfr': pfr_vec,
            'iter': np.arange(n_iter)}
)
var_names = ['Ru', 'Rc', 'Ru_mat_mixed', 'Rc_mat_mixed']
X = {}
for var in var_names:
    X[var] = Z.copy()
#X = xr.Dataset(X)

# Read iterations
for n in range(n_iter):
    print(f'Iter: {n}')
    fpath_in = dirpath_in / f'Ru_Rc_req_{n}_3.pkl'
    with open(fpath_in, 'rb') as f:
        X_ = pickle.load(f)
    X_['Ru_mat_mixed'] = X_['Ru']
    for var in var_names:
        X[var].loc[{'iter': n}] = X_[var]

# Nullify zeroth Ru and Rc for compatibility
X['Ru'].loc[{'iter': 0}] = np.nan
X['Rc'].loc[{'iter': 0}] = np.nan

X['pop_names'] = pop_names
X['pfr'] = pfr_vec

# Save the result
dirpath_out = dirpath_in.parent / 'info_new'
os.makedirs(dirpath_out, exist_ok=True)
fpath_out = dirpath_out / f'Ru_Rc_req_{n_iter-1}_3.pkl'
with open(fpath_out, 'wb') as fid:
    pickle.dump(X, fid)
