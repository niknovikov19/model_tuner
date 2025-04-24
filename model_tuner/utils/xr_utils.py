from typing import Tuple

import numpy as np
import xarray as xr


def extract_2d_points_from_xr(
        Z: xr.DataArray,
        drop_nan: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract coords and values from a 2D xarray. """
    if Z.ndim != 2:
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