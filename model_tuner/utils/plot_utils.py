from typing import List
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_xr(
        Z: xr.DataArray,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = 'viridis',
        show_ax_names: bool = True,
        colorbar: bool = True,
        xmult: int = 1,
        ymult: int = 1,
        margin: float = 0
        ) -> None:
    """Plot 2D xarray. """

    x = Z.coords[Z.dims[1]] * xmult
    y = Z.coords[Z.dims[0]] * ymult
    xx, yy = np.meshgrid(x, y)

    z = Z.values
    vmin = vmin or np.nanmin(z)
    vmax = vmax or np.nanmax(z)

    ax = plt.gca()
    mesh = ax.pcolormesh(xx, yy, z, shading='auto', cmap=cmap,
                         vmin=vmin, vmax=vmax)
    
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xmargin = (xmax - xmin) * margin
    ymargin = (ymax - ymin) * margin
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)

    if show_ax_names:
        ax.set_xlabel(Z.dims[1])
        ax.set_ylabel(Z.dims[0])
    
    if colorbar:
        plt.colorbar(mesh, ax=ax)


def plot_xr_contour(
        Z: xr.DataArray,
        name: str,
        levels: List[float],
        colors: List[str] | None = None,
        style: str = '-',
        xmult: int = 1,
        ymult: int = 1
        ) -> None:

    def fmt_func(x):
        if x == int(x): return f'{name}={int(x)}'
        else: return f'{name}={x:.1f}'
    
    x = Z.coords[Z.dims[1]] * xmult
    y = Z.coords[Z.dims[0]] * ymult

    contours = plt.contour(
        x, y, Z.values, linestyles=style,
        levels=levels, colors=colors
    )
    plt.clabel(contours, inline=True, fontsize=8, 
               fmt=fmt_func, rightside_up=True)


def set_qt_backend():
    global plt
    if 'matplotlib.pyplot' in sys.modules:
        del sys.modules['matplotlib.pyplot']  # remove old pyplot with bad backend
    matplotlib.use('Qt5Agg', force=True)  # re-set the backend
    import matplotlib.pyplot as plt_
    plt = plt_