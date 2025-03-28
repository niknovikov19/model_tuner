from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.opt.slicers import LinearSlicer
from model_tuner.utils.plot_utils import plot_xr


# Path to pre-calculated rate and CV matrices (oustd x ouamp)
#dirpath_in = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_02_28')
#fpath_in = dirpath_in / 'OUmapping_0228.pkl'
dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_03_13')
fpath_in = dirpath_base / 'OUmapping_master_compat.pkl'

# Output folder for plots
dirpath_out = Path(r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_linear_slicer_irreg')
dirpath_out.mkdir(exist_ok=True)

# Load rate and CV matrices
with open(fpath_in, 'rb') as file:
    data = pickle.load(file)

pop_names = list(data['rate'].keys())

for m, pop_name in enumerate(pop_names):

    print(f'{m} {pop_name}...')

    R = data['rate'][pop_name]
    CV = data['isicv'][pop_name]

    Xvars = {'Firing rate': R, 'CV': CV}

    # Coordinates
    ou_mean_vec = R.columns.values * 100
    ou_std_vec = R.index.values * 100

    plt.figure(111, figsize=(10, 8))
    plt.clf()
    plt.ion()

    for n, (xname, X) in enumerate(Xvars.items()):

        # Slice: ou_std = k * ou_mean
        k = 0.4

        # Convert matrix to xarray
        X_ = xr.DataArray(
            X.values,
            dims=('ou_std', 'ou_mean'),
            coords=[('ou_std', ou_std_vec), ('ou_mean', ou_mean_vec)]
        )

        # Create an object that takes 1-d slices of matrices
        slicer = LinearSlicer(
            coord_names=['ou_std', 'ou_mean'],
            coord_main='ou_mean',
            coord_coeffs={'ou_std': (k, 0)}  # ou_std = coord_main * k + 0
        )

        # Coordinates of the slice points
        slice_coords = slicer.project_1d_coords_to_nd(ou_mean_vec)
        ou_mean_slice = slice_coords['ou_mean']
        ou_std_slice = slice_coords['ou_std']

        # Get 1-d slice of the matrix using LinearSlicer
        slice_vals = slicer.get_1d_slice(X_, ou_mean_vec)

        # Get 1-d slice of the matrix using xarray interp()
        slice_coords_ = {
            'ou_mean': xr.DataArray(ou_mean_slice, dims='points'),
            'ou_std': xr.DataArray(ou_std_slice, dims='points'),
        }
        slice_vals_ = X_.interp(**slice_coords_, method='linear').values
        
        # Plot the matrix and the slice line (ou_std = k * ou_mean)
        plt.subplot(2, 2, 2 * n + 1)
        par = {}
        if xname == 'CV':
            par |= {'vmin': 0, 'vmax': 2}
        plot_xr(X_, show_ax_names=False, **par)
        plt.plot(ou_mean_slice, ou_std_slice, 'k--')
        if n == 1:  plt.xlabel('ouamp * 100')
        plt.ylabel('oustd * 100')
        plt.title(f'{pop_name}: {xname}')
        plt.xlim(ou_mean_vec[0], ou_mean_vec[-1])
        plt.ylim(ou_std_vec[0], ou_std_vec[-1])

        # Plot the slice values vs. the main coordinate (ou_mean)
        plt.subplot(2, 2, 2 * n + 2)
        plt.plot(ou_mean_slice, slice_vals, 'r', lw=1)
        plt.plot(ou_mean_slice, slice_vals_, 'k--', lw=2)
        if n == 1: plt.xlabel('ouamp * 100')
        plt.ylabel(xname)
        plt.title(f'{pop_name}: {xname}')
        if n % 2 == 1:  plt.ylim(0, 2)

    plt.draw()
    plt.show()

    fpath_out = dirpath_out / f'{m}_{pop_name}.png'
    plt.savefig(fpath_out, dpi=300)

    #break

#input('Press any key...')