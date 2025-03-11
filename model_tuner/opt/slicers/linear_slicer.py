from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from .slicer_base import Slicer


class LinearSlicer(Slicer):
    """
    Takes 1-d slices of n-dimensional data along a linear subspace.

    

    """

    def __init__(
            self,
            coord_names: List[str],
            coord_main: str,
            coord_coeffs: Dict[str, Tuple[float, float]]
            ):
        self.coord_names = coord_names
        self.coord_main = coord_main
        self.coord_coeffs = coord_coeffs

        if coord_main not in coord_names:
            raise ValueError(f'Invalid main coordinate name: {coord_main}')
        if coord_main in coord_coeffs:
            raise ValueError('Main coordinate cannot be in coord_coeffs')
        coord_coeffs[coord_main] = (1, 0)  # main coord. is projected to itself
        if set(coord_coeffs.keys()) != set(coord_names):
            raise ValueError('coord_coeffs keys must match coord_names')

    def project_1d_coords_to_nd(
            self,
            main_coord_vals: np.ndarray | float,   # value(s) of the main coordinate
            ) -> Dict[str, np.ndarray | float]:    # value(s) of every coordinate 
        res = {}
        for coord_name in self.coord_names:
            coeff = self.coord_coeffs[coord_name]
            res[coord_name] = coeff[0] * main_coord_vals + coeff[1]
        return res

    def get_1d_slice(
            self,
            X: xr.DataArray,               # n-dimensional xarray to slice from
            main_coord_vals: np.ndarray    # values of the main coordinate
            ) -> np.ndarray:               # 1-dimensional slice of X
        
        # Check the consistency of coordinates
        if set(X.coords.keys()) != set(self.coord_names):
            raise ValueError('Coordinates of X must match the object coordinates')
        
        # Get n-dimensional coordinates of the slice points
        coords = self.project_1d_coords_to_nd(main_coord_vals)
        coords = {
            name: xr.DataArray(cc, dims='points')  # to get 1-d array instead of grid
            for name, cc in coords.items()
        }

        # Get the slice
        return X.interp(**coords, method='linear').values
