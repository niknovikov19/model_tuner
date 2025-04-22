import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xarray as xr


def interpolate_to_xr(
        data_coords: List[Tuple[float, float]],  # (x, y),
        data_values: np.ndarray,
        xrange: Tuple[float, float], 
        yrange: Tuple[float, float],
        nx: int = 50,
        ny: int = 50,
        coord_names: Tuple[str, str] = ('x', 'y'),
        method: str = 'cubic'
        ) -> xr.DataArray:
    """Create 2D xarray by interpolating a list of data points. """

    data_coords = np.array(data_coords)
    mask = ~np.isnan(data_values)
    data_values = data_values[mask]
    xx_in = [c[0] for c in data_coords[mask]]
    yy_in = [c[1] for c in data_coords[mask]]

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

