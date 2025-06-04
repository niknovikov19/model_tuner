from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.opt.slicers import LinearSlicer

#from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import NetRegime1DList
from model_tuner.opt.ir_mappers import (
    PopIRMapper1DSlice,
    NetIRMapper1DSlice
)


@dataclass
class IRMapConfigRateFrom1DBatchResXR:

    pop_names: Tuple[str, ...] = ()

    # Path to the file with the batch sim result
    fpath_sim_res: str = ''

    # Parameters that corresond to the input variables
    batch_param_names: Tuple[str, str] = ('', '')

    # Parameter that was varied in the batch experiment
    batch_param_main: str = ''

    # Secondary input parameter (the other one is batch_param_main)
    @property
    def batch_param_sec(self) -> str:
        return next(
            (param for param in self.batch_param_names 
            if param != self.batch_param_main),
            None
        )

    # Relation between the main input parameter and the derived one (secondary)
    slice_method: Literal['batch_param_linear'] = 'batch_param_linear'

    # Slice parameters
    # 'batch_param_linear':
    #   batch_param_sec = batch_param_main * batch_param_ratio + batch_param_intercept
    batch_param_ratio: float = 0.4
    batch_param_intercept: float = 0.0

    # I-R mapping type and hyperparameters
    map_type: MapFuncType = MapFuncType.RICHARDS_1D  # asymmetric sigmoid
    map_params: Dict = field(default_factory=lambda: {
        'y_limits': (0, np.inf)  # output is a rate, should be non-negative
    })

    # Fitting bounds for I-R mapping parameters
    fit_param_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'q': (1, 20)  # asymmetry coefficient of RICHARDS_1D mapping
        }
    )

    # Range of the input values used for fitting
    inp_limits: Dict[str, Tuple[float, float]] | None = None

    # Weights for the fitting (prioritize the points with low rates)
    # Formula: weight = np.clip(rate ** fit_weight_pow, *fit_weight_limits)
    use_fit_weights: bool = True
    fit_weight_pow: float = 0.5
    fit_weight_limits: Tuple[float, float] = (0.1, 10)

    # Paramteres of the fitting algorithm (tolerances, etc...)
    map_fit_params: MapFitParams = MapFitParams()
    
    def __post_init__(self):
        if self.batch_param_main not in self.batch_param_names:
            raise ValueError(
                'batch_param_main should be one of batch_param_names'
            )
    
    def init_ir_mapper(
            self,
            ) -> Tuple[NetIRMapper1DSlice,
                       Dict[str, xr.DataArray]]:
        return _init_ir_mapper(self)


def _weight_func(
        x: np.ndarray,
        par_: IRMapConfigRateFrom1DBatchResXR
        ) -> np.ndarray:
    """Calculate weights for fitting based on the firing rates. """
    if not par_.use_fit_weights: return None
    x = np.clip(x, 0, np.inf)
    return np.clip(x ** par_.fit_weight_pow, *par_.fit_weight_limits)


def _init_ir_mapper(
        par: IRMapConfigRateFrom1DBatchResXR
        ) -> tuple[NetIRMapper1DSlice,
                   dict[str, xr.DataArray]]:

    # Load xarray with 1-d batch simulation results
    X = xr.load_dataset(par.fpath_sim_res)
    
    pop_names = par.pop_names

    #TODO: check par.batch_param_names and par.batch_param_main
    
    # Object that derives the secondary parameter:
    # ou_std = ou_mean * par.batch_param_ratio + par.batch_param_intercept
    slicer = LinearSlicer(
        coord_names=['ou_std', 'ou_mean'],
        coord_main='ou_mean',
        coord_coeffs={'ou_std': (par.batch_param_ratio,
                                 par.batch_param_intercept)}
    )
    
    # Network I-R mapper
    net_ir_mapper = NetIRMapper1DSlice()

    R = {}

    for pop_name in pop_names:
        print(f'Fitting I-R mapper for {pop_name}...')

        ou_mean = X['ou_mean'].sel(pop=pop_name).values

        rr = X['rate'].sel(pop=pop_name)
        rr = rr.assign_coords({'ou_mean': ('job', ou_mean)})
        rr = rr.set_index({'job': 'ou_mean'})
        rr = rr.rename({'job': 'ou_mean'})
        R[pop_name] = rr

        # TODO: exception if inp_limits is not None

        # Train pop. I-R mapper
        pop_ir_mapper = PopIRMapper1DSlice(
            slicer=slicer,
            map_type=par.map_type,
            map_params=par.map_params
        )
        pop_ir_mapper.fit_from_data_1d(
            values_in=ou_mean,
            values_out=rr.values,
            fit_params=par.map_fit_params,
            weight_func=lambda x: _weight_func(x, par),
            bounds=par.fit_param_bounds
        )

        # Add population I-R mapper to the network I-R mapper
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)
    
    return net_ir_mapper, R
