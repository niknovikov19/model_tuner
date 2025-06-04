import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from model_tuner.opt.map_funcs import MapFunc1DRichards, MapFitParams


dirpath_base = Path(r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_subnet_opt_A1')

# Experiment
exp_name = 'i_ou_wmult_0.02_5_cv'

# OU input slice
ou_std_mean_ratio = 0.2
ou_std_intercept = 0.002

# Load simulated rates
dirpath_exp = dirpath_base / exp_name
fpath_in = dirpath_exp / 'batch_result.nc'
X = xr.load_dataset(fpath_in)

# Load target rates
fpath_tbl = dirpath_exp / 'target_rates.csv'
df = pd.read_csv(fpath_tbl, index_col=0)
target_rates = {pop: df.loc[pop]['target_rate'] for pop in df.index}

# Create output folder for plots
dirpath_out = dirpath_exp / 'ir_plots_2'
os.makedirs(dirpath_out, exist_ok=True)

# OU inputs yielding the target rates
ou_inputs = {}

# Target regimes (rate, cv)
target_regimes = {}

for pop in X['pop'].values:
    print(f'Plot {pop}')

    ou_mean = X['ou_mean'].sel(pop=pop).values
    rr = X['rate'].sel(pop=pop).values
    cv = X['cv'].sel(pop=pop)

    # Fit map func to simulated data
    map_func = MapFunc1DRichards(y_limits=(0, np.inf))
    fit_par = MapFitParams()
    map_func.fit(ou_mean, rr)

    # Apply inv. map func to the target rate
    r0 = target_rates[pop]
    ou_mean_0 = map_func.apply_inv(r0)
    ou_std_0 = ou_std_intercept + ou_std_mean_ratio * ou_mean_0
    ou_inputs[pop] = {
        'ou_mean': ou_mean_0,
        'ou_std': ou_std_0
    }

    # Get CV corresponding to the target rate
    cv = cv.assign_coords({'ou_mean': ('job', ou_mean)})
    cv = cv.set_index({'job': 'ou_mean'})
    cv = cv.rename({'job': 'ou_mean'})
    cv0 = cv.interp(ou_mean=ou_mean_0)

    target_regimes[pop] = {'rate': r0, 'cv': cv0}

    # Apply map func to a fine ou_mean grid
    ou_mean_max = ou_mean.max()
    if not np.isnan(ou_mean_0):
        ou_mean_max = max(ou_mean_max, ou_mean_0)
    ou_mean_hat = np.linspace(ou_mean.min(), ou_mean_max, 200)
    rr_hat = map_func.apply(ou_mean_hat)

    # Plot
    plt.figure(111)
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(ou_mean_hat, rr_hat)
    plt.plot(ou_mean, rr, 'k.', markersize=8)
    plt.plot(ou_mean_0, r0, 'r.', markersize=10)
    plt.plot([ou_mean_hat.min(), ou_mean_hat.max()],
             [r0, r0], 'r--')
    #plt.xlabel('ou_mean')
    plt.title(f'Rate, {pop}')
    ymax = np.nanmax([rr_hat.max(), rr.max(), r0]) * 1.2
    plt.ylim(0, ymax)
    plt.xlim(ou_mean_hat.min(), ou_mean_hat.max())

    plt.subplot(2, 1, 2)
    plt.plot(ou_mean, cv.values, 'k.', markersize=8)
    plt.plot(ou_mean_0, cv0, 'r.', markersize=10)
    plt.xlabel('ou_mean')
    plt.title(f'CV, {pop}')
    plt.xlim(ou_mean_hat.min(), ou_mean_hat.max())

    # Save
    fpath_out = dirpath_out / f'{pop}.png'
    plt.savefig(fpath_out, dpi=300)

# Save ou_inputs to json
fpath_json = dirpath_exp / 'ou_inputs.json'
with open(fpath_json, 'w') as fid:
    json.dump(ou_inputs, fid, indent=4)

# Save target_regimes to CSV
fpath_csv = dirpath_exp / 'target_regimes.csv'
df_target_regimes = pd.DataFrame([
    {
        'pop_name': pop,
        'target_rate': regime['rate'],
        'target_cv': np.round(regime['cv'].values, 2)
    }
    for pop, regime in target_regimes.items()
])
df_target_regimes.to_csv(fpath_csv, index=False)
