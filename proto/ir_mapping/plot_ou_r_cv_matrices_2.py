from pathlib import Path
import pickle
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from skimage import measure
import xarray as xr

from model_tuner.utils import (
    plot_xr,
    plot_xr_contour,
    extract_2d_points_from_xr,
    interpolate_to_xr
)


#dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_02_28')
#fpath_in = dirpath_base / 'OUmapping_0228.pkl'
dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_03_13')
fpath_in = dirpath_base / 'OUmapping_master_compat.pkl'
#dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_03_26')
#fpath_in = dirpath_base / 'OUmapping_v45_batch21.pkl'

ouamp_max = 3
r_max = 50

with open(fpath_in, 'rb') as file:
    data = pickle.load(file)

#pop_names = list(data['rate'].keys())
pop_names = ['ITS4']

interactive = 1

if not interactive:
    dirpath_out = dirpath_base / f'plots_2d_xmax={ouamp_max}_rmax={r_max}_cvmode=1'
    dirpath_out.mkdir(exist_ok=True)

for m, pop_name in enumerate(pop_names):

    print(f'{m} {pop_name}...')

    R = data['rate'][pop_name]
    CV = data['isicv'][pop_name]

    Xvis = {'Rate': R, 'CV': CV}

    # Coordinates
    ouamp_vec = R.columns.values * 100
    oustd_vec = R.index.values * 100
        
    # Prepare interpolated xarrays
    Xfilled = {}
    for xname, X in Xvis.items():
        # Convert pandas 2D table to 2D xarray (irreg. grid, with nan's)
        X0_ = xr.DataArray(
            X.values,
            dims=('oustd', 'ouamp'),
            coords=[('oustd', oustd_vec), ('ouamp', ouamp_vec)]
        )
        # Extract points from the xarray (omit nan's)
        oustd_vec_, ouamp_vec_, vals = extract_2d_points_from_xr(X0_)
        # Interpolate the points to fill a dense regular xarray
        Xfilled[xname] = interpolate_to_xr(
            data_coords=list(zip(ouamp_vec_, oustd_vec_)),
            data_values=vals,
            nx=500, ny=500,
            coord_names=('ou_mean', 'ou_std')            
        )
    
    plt.ion()
    plt.figure(111)
    plt.clf()
    
    for n, (xname, X) in enumerate(Xvis.items()):       
        X_ = Xfilled[xname]        
        plt.subplot(1, 2, n + 1)
        
        par = {}
        if xname == 'CV':
            par |= {'vmin': 0, 'vmax': 2}
        
        plot_xr(X_, show_ax_names=False, **par)

        plot_xr_contour(Xfilled['Rate'], 'r', levels=[5, 10],
                        colors=['r', 'k'], style='-')
        plot_xr_contour(Xfilled['CV'], 'CV', levels=[0.5, 1],
                        colors=['r', 'k'], style='--')
        
        #plt.legend()
        plt.xlabel('ouamp * 100')
        plt.ylabel('oustd * 100')
        plt.title(f'{pop_name}: {xname}')
        plt.xlim(ouamp_vec[0], ouamp_vec[-1])
        plt.ylim(oustd_vec[0], oustd_vec[-1])

    if interactive:
        plt.draw()
        plt.show()
        break
    else:
        fpath_out = dirpath_out / f'{m}_{pop_name}.png'
        plt.savefig(fpath_out, dpi=300)

if interactive:
    input('Press any key...')