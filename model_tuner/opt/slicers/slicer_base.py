from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import xarray as xr


class Slicer(ABC):

    def __init__(
            self,
            coord_names: List[str],
            coord_main: str,
            ):
        self.coord_names = coord_names
        self.coord_main = coord_main

    @abstractmethod
    def project_1d_coords_to_nd(
            self,
            main_coord_vals: np.ndarray | float,   # value(s) of the main coordinate
            ) -> Dict[str, np.ndarray | float]:    # value(s) of every coordinate 
        pass
    
    @abstractmethod
    def get_1d_slice(
            self,
            X: xr.DataArray,               # n-dimensional xarray to slice from
            main_coord_vals: np.ndarray    # values of the main coordinate
            ) -> np.ndarray:               # 1-dimensional slice of X
        pass