from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
from typing import Dict, List, Literal, Tuple

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

from model_tuner.utils import plot_xr, interpolate_to_xr

from .read_batch_res_table_ import read_batch_res_table


@dataclass
class IRMapConfigRateFrom2DBatchResLists:

    pop_names: Tuple[str, ...] = ()

    # Path(s) to csv file(s) with batch sim results
    # (each row contains two batch coords and rate/CV for every pop.)
    fpath_sim_res: str | List[str]= ''

    # Parameters that were varied in the batch experiment
    batch_param_names: Tuple[str, str] = ('', '')

    # Batch parameter that parametrizes 1-d slice of 2-d matrix
    batch_param_main: str = ''

    # Grid to interpolate sim results
    # {batch_param: (min, max, npoints)}
    batch_param_grid: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)

    # Secondary batch parameter (the other one is batch_param_main)
    @property
    def batch_param_sec(self) -> str:
        return next(
            (param for param in self.batch_param_names 
            if param != self.batch_param_main),
            None
        )

    # Method of slicing the firing rate matrix to get 1-d data
    # 'batch_param_linear' - slice along the line with a linear
    #    relation between batch parameters
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

    # Range of the input values (batch_param_main) used for fitting
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
        par_: IRMapConfigRateFrom2DBatchResLists
        ) -> np.ndarray:
    """Calculate weights for fitting based on the firing rates. """
    if not par_.use_fit_weights: return None
    x = np.clip(x, 0, np.inf)
    return np.clip(x ** par_.fit_weight_pow, *par_.fit_weight_limits)


def _init_ir_mapper(
        par: IRMapConfigRateFrom2DBatchResLists,
        ) -> Tuple[NetIRMapper1DSlice,
                   Dict[str, xr.DataArray]]:

    # Load simulation results:
    # accumulate (ou_mean, ou_std, rate, CV) points for each population
    # by reading simulation result CSV files
    fpaths_in = par.fpath_sim_res
    fpaths_in = fpaths_in if isinstance(fpaths_in, list) else [fpaths_in]
    sim_data = {}
    for pop_name in par.pop_names:
        sim_data[pop_name] = {
            v: [] for v in ['ou_mean', 'ou_std', 'rate', 'CV']
        }
    for fpath_in in fpaths_in:
        print(f'Reading {fpath_in}...')
        ou_mean, ou_std, data = read_batch_res_table(fpath_in)
        for pop_name, pop_data in data.items():
            pop_data_ = sim_data[pop_name]
            pop_data_['ou_mean'] += list(ou_mean)
            pop_data_['ou_std'] += list(ou_std)
            pop_data_['rate'] += list(pop_data['Rate'])
            pop_data_['CV'] += list(pop_data['CV'])
    
    # Convert firing rate matrices to xr DataArrays
    R = {}
    grid = par.batch_param_grid
    for pop_name, pop_data in sim_data.items():
        print(f'Interpolating data for {pop_name}...')
        R[pop_name] = interpolate_to_xr(
            data_coords=list(zip(pop_data['ou_mean'],
                                 pop_data['ou_std'])),
            data_values=pop_data['rate'],
            xrange=(grid['ou_mean'][0],
                    grid['ou_mean'][1]),
            yrange=(grid['ou_std'][0],
                    grid['ou_std'][1]),
            nx=grid['ou_mean'][2],
            ny=grid['ou_std'][2],
            coord_names=('ou_mean', 'ou_std')
        )

    # Object that takes 1-d slices of the firing rate matrices
    # Slice: ou_std = ou_mean * par.batch_param_ratio + par.batch_param_intercept
    slicer = LinearSlicer(
        coord_names=['ou_std', 'ou_mean'],
        coord_main='ou_mean',
        coord_coeffs={'ou_std': (par.batch_param_ratio,
                                 par.batch_param_intercept)}
    )
    
    # Network I-R mapper
    net_ir_mapper = NetIRMapper1DSlice()

    for pop_name in par.pop_names:
        print(f'Fitting I-R mapper for {pop_name}...')
        
        # Firing rate matrix of a population (ou_std x ou_mean)
        if pop_name not in R:
            raise ValueError(f'Simulation data for {pop_name} '
                             'not found in the CSV file(s)')
        R_ = R[pop_name]
        
        # Select submatrix of R_ with the main coordinate within the range
        ou_mean_min, ou_mean_max = par.inp_limits[pop_name]
        mask = (R_.ou_mean >= ou_mean_min) & (R_.ou_mean <= ou_mean_max)
        R_ = R_.where(mask, drop=True)

        # Train pop. I-R mapper on a slice of the firing rate matrix
        pop_ir_mapper = PopIRMapper1DSlice(
            slicer=slicer,
            map_type=par.map_type,
            map_params=par.map_params
        )
        pop_ir_mapper.fit_from_data(
            X=R_,
            fit_params=par.map_fit_params,
            weight_func=lambda x: _weight_func(x, par),
            bounds=par.fit_param_bounds
        )

        # Add population I-R mapper to the network I-R mapper
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)
    
    return net_ir_mapper, R
