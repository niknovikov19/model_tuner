from typing import List, Tuple

import numpy as np
from scipy.interpolate import griddata
import xarray as xr

from .xr_utils import extract_2d_points_from_xr


def interpolate_to_xr(
        data_coords: List[Tuple[float, float]],  # (x, y),
        data_values: np.ndarray,
        xrange: Tuple[float, float] | None = None, 
        yrange: Tuple[float, float] | None = None,
        nx: int = 50,
        ny: int = 50,
        coord_names: Tuple[str, str] = ('x', 'y'),
        method: str = 'cubic'
        ) -> xr.DataArray:
    """Create 2D xarray by interpolating a list of data points. """

    data_coords = np.array(data_coords)
    data_values = np.array(data_values)
    mask = ~np.isnan(data_values)
    data_values = data_values[mask]
    xx_in = [c[0] for c in data_coords[mask]]
    yy_in = [c[1] for c in data_coords[mask]]

    xrange = xrange or (min(xx_in), max(xx_in))
    yrange = yrange or (min(yy_in), max(yy_in))

    xx_out = np.linspace(xrange[0], xrange[1], nx)
    yy_out = np.linspace(yrange[0], yrange[1], ny)
    y_grid, x_grid = np.meshgrid(yy_out, xx_out, indexing='ij')
    
    values_interp = griddata(
        (yy_in, xx_in),        # known coordinates
        data_values,           # known values
        (y_grid, x_grid),      # target grid
        method=method
    )
    return xr.DataArray(
        values_interp,
        dims=(coord_names[1], coord_names[0]),
        coords=[(coord_names[1], yy_out),
                (coord_names[0], xx_out)]
    )

def interp_points_from_2d_xr(
        Z: xr.DataArray,
        y: np.ndarray,
        x: np.ndarray,
        method: str = 'cubic'
        ) -> np.ndarray:
    """Interpolate 2D xarray at given points. """
    yy, xx, zz = extract_2d_points_from_xr(Z, drop_nan=True)
    zz_ = griddata(
        (yy, xx),        # known coordinates
        zz,              # known values
        (y, x),          # target points
        method=method
    )
    return zz_