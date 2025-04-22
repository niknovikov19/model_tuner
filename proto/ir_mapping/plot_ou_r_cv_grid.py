import os
from pathlib import Path
#from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import griddata
from skimage import measure
import xarray as xr

from model_tuner.utils import plot_xr, interpolate_to_xr
from read_batch_res_table import read_batch_res_table


# Contour function: y = k * x
def contour_func_const_ratio(x, y, k):
    return y - k * x

# Find contours: func(x, y) = level
def find_contours(func, x, y, level=0, **kwargs):
    xx, yy = np.meshgrid(x, y)
    z = func(xx, yy, **kwargs)
    C_idx = measure.find_contours(z, level)
    C = []
    for n, c_idx in enumerate(C_idx):
        c_idx = c_idx.round().astype(int)
        c = np.full_like(c_idx, np.nan, dtype=float)
        c[:, 0] = y[c_idx[:, 0]]
        c[:, 1] = x[c_idx[:, 1]]
        C.append(c)
        C_idx[n] = c_idx
    return C, C_idx


dirpath_base = Path('D:\\WORK\\Salvador\\repo\\A1_OUinp\\exp_results')

exp_name = 'batch_ougrid_vip_0'

grid_sz = 150
margin = 0.05
coord_mult = 100

mode = 'slices'
#mode = 'contours'

# Slices: (k, c), such that ou_std = ou_mean * k + c
slices = [(0.4, c) for c in [0, 1, 2]]


# Experiment results fodler
dirpath_exp = dirpath_base / exp_name

# Output foler for saving plots
dirpath_out = dirpath_exp / 'plots_other' / f'{mode}'
os.makedirs(dirpath_out, exist_ok=True)

# Read exp results table
fpath_in = dirpath_exp / 'batch_result.csv'
ou_mean, ou_std, data = read_batch_res_table(fpath_in)
ou_mean *= coord_mult
ou_std *= coord_mult
pop_names = list(data.keys())

# Interpolate data to fill the visualized grid
data_interp = {}
for pop in pop_names:
    data_interp[pop] = {}
    for data_type in ('Rate', 'CV'):
        data_interp[pop][data_type] = interpolate_to_xr(
            data_coords=list(zip(ou_mean, ou_std)),
            data_values=data[pop][data_type],
            xrange=(ou_mean.min(), ou_mean.max()),
            yrange=(ou_std.min(), ou_std.max()),
            nx=grid_sz,
            ny=grid_sz,
            coord_names=('ou_mean', 'ou_std')
        )

def plot_pop_data(pop: str):

    for n, data_type in enumerate(['Rate', 'CV']):

        Z = data_interp[pop][data_type]
        ou_mean_ = Z.coords['ou_mean'].values
        ou_std_ = Z.coords['ou_std'].values

        # 2D image with horiontal slicing lines
        nx, ny = 2, 2
        plt.subplot(ny, nx, n * nx + 1)
        plot_xr(Z, margin=margin, show_ax_names=False)
        for slice in slices:
            x1, x2 = ou_mean_.min(), ou_mean_.max()
            y1 = slice[0] * x1 + slice[1]
            y2 = slice[0] * x2 + slice[1]
            plt.plot((x1, x2), (y1, y2), '--')
        plt.title(f'{data_type}, {pop}')
        if n == ny - 1:
            plt.xlabel(f'ou_mean * {coord_mult}')
        plt.ylabel(f'ou_std * {coord_mult}')
        
        # 1D slices: value vs. ou_mean (ou_std==const)
        plt.subplot(ny, nx, n * nx + 2)
        for slice in slices:
            ou_std_slice_ = ou_mean_ * slice[0] + slice[1]
            slice_data = Z.interp(
                ou_mean=xr.DataArray(ou_mean_, dims='points'),
                ou_std=xr.DataArray(ou_std_slice_, dims='points'),
            )
            plt.plot(ou_mean_, slice_data.values)
        plt.title(f'{data_type} slices (ou_std=const), {pop}')
        if n == ny - 1:
            plt.xlabel(f'ou_mean * {coord_mult}')
        plt.ylabel(data_type)

#plt.ion()
plt.figure(figsize=(14, 7))

for pop in pop_names:
    print(f'Plotting {pop}...')

    plt.clf()
    plot_pop_data(pop)

    fpath_fig = dirpath_out / f'{pop}.png'
    plt.savefig(fpath_fig, dpi=300)

#plt.show()
#input('Press any key...')
