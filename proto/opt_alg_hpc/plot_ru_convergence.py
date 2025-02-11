from glob import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


dirpath_in = Path(
    r'D:\WORK\Salvador\repo\model_tuner\proto\opt_alg_hpc\data\test_opt_hpc_batch'
    r'\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.25_alpha=0.25\info'
    #r'\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.1_alpha=0.5\info'
    #r'\exp_r0=(2_10_5_15)_pfr=(0.1_1.5_7)_wmult=0.1_alpha=0.25\info'
)

n_iter = 50
n_pop = 4
n_pts = 7

pop_names = ['L2e', 'L2i', 'L4e', 'L4i']

Ru = np.full((n_pop, n_pts, n_iter), np.nan)

for n in range(n_iter):
    fpath_mask = str(dirpath_in / f'Ru_Rc_req_{n}_*.pkl')
    fpath_in = glob(fpath_mask)[0]
    with open(fpath_in, 'rb') as fid:
        res = pickle.load(fid)
    n_pts_ = res['Ru'].shape[1]
    Ru[:, :n_pts_, n] = res['Ru']

plt.figure()
for pop_num in range(n_pop):
    plt.subplot(2, 2, pop_num + 1)
    for n in range(n_pts):
        rr = Ru[pop_num, n, :]
        plt.plot(rr)
        plt.title(pop_names[pop_num])


    