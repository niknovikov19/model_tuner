from glob import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


dirpath_in = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_opt_L24_hpc_batch_qsub'
    r'\experiments'
    r'\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.25_alpha=0.25\info'
)

n_iter = 10
n_pop = 4
n_pts = 7

pop_names = ['L2e', 'L2i', 'L4e', 'L4i']

rr_base={
    'L2e': 2.,
    'L2i': 10.,
    'L4e': 5.,
    'L4i': 15.
}

Ru = np.full((n_pop, n_pts, n_iter), np.nan)
Rc = np.full((n_pop, n_pts, n_iter), np.nan)

for n in range(n_iter):
    fpath_mask = str(dirpath_in / f'Ru_Rc_req_{n}_*.pkl')
    fpaths = glob(fpath_mask)
    #print(f'Files: {fpaths}')
    if len(fpaths) == 0:
        raise RuntimeError(f'No files matching: {fpath_mask}')
    fpath_in = fpaths[0]
    with open(fpath_in, 'rb') as fid:
        res = pickle.load(fid)
    n_pts_ = res['Ru'].shape[1]
    Ru[:, :n_pts_, n] = res['Ru']
    Rc[:, :n_pts_, n] = res['Rc']

plt.ion()
plt.figure()
for pop_num, pop_name in enumerate(pop_names):
    plt.subplot(2, 2, pop_num + 1)
    for n in range(n_pts):
        rr = Rc[pop_num, n, :]
        plt.plot(rr, '.-')
    plt.plot([0, n_iter], [rr_base[pop_name]] * 2, 'k--')
    plt.title(pop_name)

plt.figure()
for pop_num, pop_name in enumerate(pop_names):
    plt.subplot(2, 2, pop_num + 1)
    rmax = Ru[pop_num, :, :].ravel().max()
    plt.plot([0, rmax], [rr_base[pop_name]] * 2, 'k--')
    plt.plot(Ru[pop_num, -3, :], Rc[pop_num, -3, :], 'k-')
    for n in range(n_iter):
        ru = Ru[pop_num, :, n]
        rc = Rc[pop_num, :, n]
        plt.plot(ru, rc, '.-')
    plt.xlabel('Ru')
    plt.ylabel('Rc')
    plt.title(pop_name)

input('Press Enter to exit')
    