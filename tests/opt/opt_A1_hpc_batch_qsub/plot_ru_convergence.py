from glob import glob
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from model_tuner.utils import load_yaml
from model_tuner.main import OptExperimentParams


dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main'
    r'\test_opt_A1_hpc_batch_qsub\experiments'
    #r'\test_1_pfr=(0.4_1.2_5)_unconn_alpha=0.25'
    #r'\test_1_pfr=(0.4_1.2_5)_wmult=0.01_alpha=0.25'
    r'\test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=0.2'
)

n_iter = 20

need_plot_conv = 1
need_plot_ucfit = 1

# Load experiment params
fpath_exp_params = dirpath_base / 'exp_params.yaml'
exp_params = load_yaml(fpath_exp_params,
                          data_class=OptExperimentParams)

# Load target firing rates
df_target_rates = pd.read_csv(dirpath_base / 'target_rates.csv')
target_rates = dict(zip(df_target_rates['pop_name'],
                        df_target_rates['target_rate']))

pop_names = list(exp_params.pop_names)
pfr_vec = exp_params.pfr_vec

print(type(pfr_vec))

# Output folder
dirpath_figs_conv = dirpath_base / 'convergence_figs'
dirpath_figs_ucfit = dirpath_base / 'uc_fit_figs'
os.makedirs(dirpath_figs_conv, exist_ok=True)
os.makedirs(dirpath_figs_ucfit, exist_ok=True)

n_pop = len(pop_names)
n_pfr = len(pfr_vec)

# Create xarray dataset to store Ru and Rc matrices
X = xr.Dataset(
    {
        "Ru": (["pop", "iter", "pfr"], np.full((n_pop, n_iter, n_pfr), np.nan)),
        "Rc": (["pop", "iter", "pfr"], np.full((n_pop, n_iter, n_pfr), np.nan)),
    },
    coords={"pop": pop_names, "iter": range(n_iter), "pfr": pfr_vec},
)

# Load Ru and Rc matrices
iter_data = {}
for n in range(n_iter):
    fpath_mask = str(dirpath_base / 'info' / f'Ru_Rc_req_{n}_*.pkl')
    fpath_iter_data = glob(fpath_mask)[0]
    with open(fpath_iter_data, 'rb') as f:
        iter_data = pickle.load(f)
    coords_ = {'pop': pop_names, 'pfr': iter_data['pfr']}
    Ru_ = xr.DataArray(iter_data['Ru'], dims=['pop', 'pfr'], coords=coords_)
    Rc_ = xr.DataArray(iter_data['Rc'], dims=['pop', 'pfr'], coords=coords_)
    X['Ru'].loc[{'iter': n, 'pfr': Ru_.pfr}] = Ru_
    X['Rc'].loc[{'iter': n, 'pfr': Rc_.pfr}] = Rc_

# Plot Rc vs. iter number for each population
if need_plot_conv:
    for n, pop_name in enumerate(pop_names):
        print(f'Rc vs. iter: {pop_name}')
        plt.figure(111)
        plt.clf()
        for m, pfr in enumerate(pfr_vec):
            plt.plot(
                X.coords['iter'],
                X['Rc'].loc[pop_name, :, pfr],
                '.-',
                label=f'pfr={pfr:.3f}'
            )
        plt.plot(
            [X.coords['iter'].min(), X.coords['iter'].max()],
            [target_rates[pop_name], target_rates[pop_name]],
            'k--'
        )
        plt.plot()
        plt.xlabel('Iteration')
        plt.ylabel('Rc')
        plt.title(pop_name)    
        plt.legend(loc='lower right')
        #plt.draw()
        #plt.show()
        plt.savefig(dirpath_figs_conv / f'{n}_{pop_name}.png')

# Plot Rc vs. Ru for each population
if need_plot_ucfit:
    for n, pop_name in enumerate(pop_names):
        print(f'Rc vs. Ru: {pop_name}')
        plt.figure(111)
        plt.clf()
        for m in range(n_iter):
            plt.plot(
                X['Ru'].loc[pop_name, m, :],
                X['Rc'].loc[pop_name, m, :],
                '.-'
            )
        ru_ = X['Ru'].loc[pop_name, :, :].values.ravel()
        ru_min, ru_max = np.nanmin(ru_), np.nanmax(ru_)
        r0 = target_rates[pop_name]
        plt.plot([ru_min, ru_max], [r0, r0], 'k--')
        plt.plot([ru_min, ru_max], [ru_min, ru_max], 'k--')
        plt.xlabel('Ru')
        plt.ylabel('Rc')
        plt.title(pop_name)
        plt.savefig(dirpath_figs_ucfit / f'{n}_{pop_name}.png')

input('Press Enter to exit')
    