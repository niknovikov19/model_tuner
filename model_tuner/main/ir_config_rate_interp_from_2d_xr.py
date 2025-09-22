from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.opt.slicers import LinearSlicer

#from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import NetRegime1DList
from model_tuner.opt.ir_mappers import (
    PopIRMapper1DInterp,
    NetIRMapper1DInterp
)

from model_tuner.utils import plot_xr, interpolate_to_xr

from .read_batch_res_table_ import read_batch_res_table


@dataclass
class IRMapConfigRateInterpFrom2DXR:

    pop_names: Tuple[str, ...] = ()

    # Path(s) to netcdf file(s) with batch sim results
    fpath_sim_res: str | List[str]= ''

    # Parameters that were varied in the batch experiment
    batch_param_names: Tuple[str, str] = ('', '')

    # Smoothing kernel width for each data type
    smooth_sigma: dict = field(default_factory=dict)
    
    def init_ir_mapper(
            self,
            ) -> Tuple[NetIRMapper1DInterp,
                       Dict[str, xr.DataArray]]:
        return _init_ir_mapper(self)

def _init_ir_mapper(
        par: IRMapConfigRateInterpFrom2DXR,
        ) -> Tuple[NetIRMapper1DInterp,
                   Dict[str, xr.DataArray]]:
    
    # Load batch results
    fpaths_in = par.fpath_sim_res
    fpaths_in = fpaths_in if isinstance(fpaths_in, list) else [fpaths_in]
    X = []
    for fpath in fpaths_in:
        X.append(xr.load_dataset(fpath))
    X = xr.concat(X, dim='pop', data_vars='all', coords='minimal',
                  compat='no_conflicts', join='exact')
    
    # Network I-R mapper
    net_ir_mapper = NetIRMapper1DInterp()

    R_all = {}
    for pop in X.pop.values:

        # Extract variables, assign (ou_mean, ou_std) coords
        vars = ['rate', 'cv', 'v_med_min', 'v_med_max']
        D = {}
        for v in vars:
            X_ = X[v].sel(pop=pop)
            X_ = X_.rename(ou_mean_ind='ou_mean', ou_std_ind='ou_std')
            X_ = X_.assign_coords(
                ou_mean=X['ou_mean'].sel(pop=pop).values,
                ou_std=X['ou_std'].sel(pop=pop).values
            )
            D[v] = X_.T
        
        # Smooth data
        for v, sigma in par.smooth_sigma.items():
            D[v] = xr.DataArray(
                gaussian_filter(D[v].values, sigma=sigma),
                dims=D[v].dims,
                coords=D[v].coords
            )

        # Upsample data
        sz_new = 100
        ou_mean = D['rate'].ou_mean.values
        ou_std = D['rate'].ou_std.values
        ou_mean_new = np.linspace(ou_mean.min(), ou_mean.max(), sz_new)
        ou_std_new = np.linspace(ou_std.min(), ou_std.max(), sz_new)
        for v, X_ in D.items():
            D[v] = X_.interp(ou_mean=ou_mean_new, ou_std=ou_std_new, method='linear')
        
        R_all[pop] = D['rate']

        # Train pop. I-R mapper on a slice of the firing rate matrix
        pop_ir_mapper = PopIRMapper1DInterp()
        pop_ir_mapper.set_data(D)

        # Add population I-R mapper to the network I-R mapper
        net_ir_mapper.set_pop_mapper(pop, pop_ir_mapper)
    
    return net_ir_mapper, R_all
