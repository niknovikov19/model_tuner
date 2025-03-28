import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_xr(
        Z: xr.DataArray,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = 'viridis',
        show_ax_names: bool = True,
        colorbar: bool = True
        ) -> None:
    # Plot 2D xarray
    y, x = Z[Z.dims[0]], Z[Z.dims[1]]
    xx, yy = np.meshgrid(x, y)
    z = Z.values
    vmin = vmin or np.nanmin(z)
    vmax = vmax or np.nanmax(z)
    ax = plt.gca()
    mesh = ax.pcolormesh(xx, yy, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    if show_ax_names:
        ax.set_xlabel(Z.dims[1])
        ax.set_ylabel(Z.dims[0])
    if colorbar:
        plt.colorbar(mesh, ax=ax)