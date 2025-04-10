from pathlib import Path
import pickle
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from skimage import measure
import xarray as xr


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

def get_wrapped_cmap(base_cmap='hsv', cycles=3):
    # Repeat a base colormap multiple times
    base = plt.get_cmap(base_cmap)
    colors = base(np.linspace(0, 1, 256))
    repeated = np.tile(colors, (cycles, 1))
    return mcolors.ListedColormap(repeated)

def plot_xr(Z, vmin=None, vmax=None, cmap='viridis', show_ax_names=True):
    # Plot 2D xarray
    y, x = Z[Z.dims[0]], Z[Z.dims[1]]
    xx, yy = np.meshgrid(x, y)
    z = Z.values
    vmin = vmin or np.nanmin(z)
    vmax = vmax or np.nanmax(z)
    if cmap == 'wrapped':
        cmap = get_wrapped_cmap()
    ax = plt.gca()
    mesh = ax.pcolormesh(xx, yy, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    if show_ax_names:
        ax.set_xlabel(Z.dims[1])
        ax.set_ylabel(Z.dims[0])
    plt.colorbar(mesh, ax=ax)

def extract_2d_points_from_xr(
        Z: xr.DataArray,
        drop_nan: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract coords and values from a 2D xarray
    if X.ndim != 2:
        raise ValueError('Input xarray must be 2D')
    mask = ~np.isnan(Z.values)
    y, x = Z[Z.dims[0]], Z[Z.dims[1]]
    yy, xx = np.meshgrid(y, x, indexing='ij')
    zz = Z.values
    if drop_nan:
        yy = yy[mask]
        xx = xx[mask]
        zz = zz[mask]
    return yy, xx, zz

def fill_2d_xr(
        Z: xr.DataArray,
        method: str = 'cubic'
        ) -> xr.DataArray:
    # Fill 2D xarray using SciPy's griddata
    yy, xx, zz = extract_2d_points_from_xr(Z, drop_nan=True)
    yy_, xx_, _ = extract_2d_points_from_xr(Z, drop_nan=False)
    zz_ = griddata(
        (yy, xx),        # known coordinates
        zz,              # known values
        (yy_, xx_),      # target grid
        method=method
    )
    return xr.DataArray(zz_, coords=Z.coords, dims=Z.dims)

def interp_from_2d_xr(
        Z: xr.DataArray,
        y: np.ndarray,
        x: np.ndarray,
        method: str = 'cubic'
        ) -> np.ndarray:
    # Interpolate 2D xarray at given points
    yy, xx, zz = extract_2d_points_from_xr(Z, drop_nan=True)
    zz_ = griddata(
        (yy, xx),        # known coordinates
        zz,              # known values
        (y, x),          # target points
        method=method
    )
    return zz_


#dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_02_28')
#fpath_in = dirpath_base / 'OUmapping_0228.pkl'
#dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_03_13')
#fpath_in = dirpath_base / 'OUmapping_master_compat.pkl'
dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_03_26')
fpath_in = dirpath_base / 'OUmapping_v45_batch21.pkl'

ouamp_max = 3
r_max = 50

dirpath_out = dirpath_base / f'plots_2d_xmax={ouamp_max}_rmax={r_max}_cvmode=1'
dirpath_out.mkdir(exist_ok=True)

with open(fpath_in, 'rb') as file:
    data = pickle.load(file)

pop_names = list(data['rate'].keys())
#pop_names = ['IT3']

for m, pop_name in enumerate(pop_names):

    print(f'{m} {pop_name}...')

    R = data['rate'][pop_name]
    CV = data['isicv'][pop_name]

    Xvis = {'Rate': R, 'CV': CV}

    # Coordinates
    ouamp_vec = R.columns.values * 100
    oustd_vec = R.index.values * 100
        
    # Pandas -> xarray
    Xfilled = {}
    for xname, X in Xvis.items():
        X0_ = xr.DataArray(
            X.values,
            dims=('oustd', 'ouamp'),
            coords=[('oustd', oustd_vec), ('ouamp', ouamp_vec)]
        )
        try:
            Xfilled[xname] = fill_2d_xr(X0_)
        except Exception as e:
            print(f'Exception in fill_2d_xr(): {e}')
            Xfilled[xname] = xr.full_like(X0_, np.nan)
            continue
    
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
        
        def contour_fmt_r(x):
            if x == int(x): return f'r={int(x)}'
            else: return f'r={x:.1f}'
        def contour_fmt_cv(x):
            if x == int(x): return f'CV={int(x)}'
            else: return f'CV={x:.1f}'
        
        contours_cv = plt.contour(
            ouamp_vec, oustd_vec, Xfilled['CV'], linestyles='--',
            #levels=[0.5, 1, 1.5], colors=['r', 'k', 'm']
            levels=[0.5, 1], colors=['r', 'k']
        )
        plt.clabel(contours_cv, inline=True, fontsize=8, fmt=contour_fmt_cv, rightside_up=True)
        
        contours_r = plt.contour(
            ouamp_vec, oustd_vec, Xfilled['Rate'], levels=[5, 10],
            colors=['k', 'r'], linestyles='-')
        plt.clabel(contours_r, inline=True, fontsize=8, fmt=contour_fmt_r, rightside_up=True)
        
        #plt.legend()
        plt.xlabel('ouamp * 100')
        plt.ylabel('oustd * 100')
        plt.title(f'{pop_name}: {xname}')
        plt.xlim(ouamp_vec[0], ouamp_vec[-1])
        plt.ylim(oustd_vec[0], oustd_vec[-1])

    #plt.draw()
    #plt.show()

    fpath_out = dirpath_out / f'{m}_{pop_name}.png'
    plt.savefig(fpath_out, dpi=300)

    #break

#input('Press any key...')