from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.opt.regimes import (
    PopRegime1D, NetRegime1D)
from model_tuner.opt.inputs import (
    PopInputND, NetInputND)
from model_tuner.opt.ir_mappers import (
    PopIRMapper1DInterp, NetIRMapper1DInterp)
from model_tuner.utils import (
    load_yaml, set_qt_backend,
    plot_xr, plot_xr_contour
)


def _plot_ir_ougrid_interp_pop(
        pop_ir_mapper: PopIRMapper1DInterp,
        R: PopRegime1D,
        pop: str
        ):

    D = pop_ir_mapper.maps2d
    r0 = R.value

    # Select (ou_mean, ou_std) point that maps onto R
    I = pop_ir_mapper.R_to_I(R)
    
    # Voltage levels to plot
    if pop_ir_mapper.interp_method == 'vmin_eq':
        vmin_vis = np.sort(pop_ir_mapper.interp_params['vmin_vals'])
    else:
        vmin_vis = [-150]
    
    for n, (v, X_) in enumerate(D.items()):
        plt.subplot(2, 2, n + 1)
        # Rate/CV/voltage 2-d maps
        plot_xr(X_, show_ax_names=True)
        # Rate/CV/voltage levels
        plot_xr_contour(D['rate'], '', [r0], colors=['r'])
        plot_xr_contour(D['cv'], 'cv', [1], colors=['k'])
        plot_xr_contour(D['v_med_min'], 'vmin', vmin_vis, colors=['m'])
        # Selected point
        plt.plot(I['ou_mean'], I['ou_std'], 'k.', markersize=10)
        plt.title(f'{pop} {v}')

def plot_ir_ougrid_interp(
        net_ir_mapper: NetIRMapper1DInterp,
        R: NetRegime1D,
        dirpath_out: str | Path
        ):
    dirpath_out = Path(dirpath_out)
    dirpath_out.mkdir(parents=True, exist_ok=True)
    set_qt_backend()
    plt.ion()
    plt.figure(111, figsize=(10, 8))
    for pop in R.get_pop_names():
        plt.clf()
        pop_ir_mapper = net_ir_mapper[pop]
        _plot_ir_ougrid_interp_pop(pop_ir_mapper, R[pop], pop)
        plt.draw()
        plt.show()
        fpath_out = dirpath_out / f'{pop}.png'
        plt.savefig(fpath_out)
