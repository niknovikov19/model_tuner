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
class IRMapConfigRateFrom2DRateCVMats:

    pop_names: Tuple[str, ...] = ()

    # Path to the file with the matrices of firing rates and CVs
    fpath_mats: str = ''

    # Parameters that were varied in the batch experiment
    batch_param_names: Tuple[str, str] = ('', '')

    # Batch parameter that parametrizes 1-d slice of 2-d matrix
    batch_param_main: str = ''

    # Secondary batch parameter (the other one is batch_param_main)
    @property
    def batch_param_sec(self) -> str:
        return next(
            (param for param in self.batch_param_names 
            if param != self.batch_param_main),
            None
        )

    # Method of slicing the firing rate matrix to get 1-d data
    # 'batch_param_ratio' - slice along the line with a fixed ratio
    #    of the batch parameters (batch_param_sec / batch_param_main)
    slice_method: Literal['batch_param_ratio'] = 'batch_param_ratio'

    # Ratio of the secondary batch parameter to the main one
    batch_param_ratio: float = 0.4

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
            dirpath_plots_out: Path | str | None = None
            ) -> NetIRMapper1DSlice:
        return _init_ir_mapper(self, dirpath_plots_out)


def _weight_func(
        x: np.ndarray,
        par_: IRMapConfigRateFrom2DRateCVMats
        ) -> np.ndarray:
    """Calculate weights for fitting based on the firing rates. """
    if not par_.use_fit_weights: return None
    x = np.clip(x, 0, np.inf)
    return np.clip(x ** par_.fit_weight_pow, *par_.fit_weight_limits)


def _init_ir_mapper(
        par: IRMapConfigRateFrom2DRateCVMats,
        dirpath_plots_out: Path | str | None = None
        ) -> NetIRMapper1DSlice:

    # Load rate and CV matrices
    with open(par.fpath_mats, 'rb') as file:
        mats = pickle.load(file)
    
    pop_names = par.pop_names
    
    # Convert firing rate matrices to xr DataArrays
    R = {}
    for pop_name in pop_names:
        R_ = mats['rate'][pop_name]
        ou_mean_vec = R_.columns.values
        ou_std_vec = R_.index.values
        R[pop_name] = xr.DataArray(
            R_.values,
            dims=('ou_std', 'ou_mean'),
            coords=[('ou_std', ou_std_vec), ('ou_mean', ou_mean_vec)]
        )

    # Object that takes 1-d slices of the firing rate matrices
    # Slice: ou_std = ou_mean * par.batch_param_ratio
    slicer = LinearSlicer(
        coord_names=['ou_std', 'ou_mean'],
        coord_main='ou_mean',
        coord_coeffs={'ou_std': (par.batch_param_ratio, 0)}
    )
    
    # Network I-R mapper
    net_ir_mapper = NetIRMapper1DSlice()

    for pop_name in pop_names():
        print(f'Fitting I-R mapper for {pop_name}...')
        
        # Firing rate matrix of a population (ou_std x ou_mean)
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

    if dirpath_plots_out:
        _plot_ir_mapping(
            net_ir_mapper, R, par.inp_limits,
            dirpath_out=dirpath_plots_out
        )
    
    return net_ir_mapper


def _plot_ir_mapping(
        net_ir_mapper: NetIRMapper1DSlice,
        rate_mats: Dict[str, xr.DataArray],
        inp_limits: Dict[str, Tuple[float, float]],
        dirpath_out: Path | str,
        r_vis_max: float = 100,
        inp_vis_max: float = 5,
        npts: int = 100
        ) -> None:
    
    pop_names = net_ir_mapper.pop_names
    npops = len(pop_names)
    
    # Create output folder
    if isinstance(dirpath_out, str):
        dirpath_out = Path(dirpath_out)
    os.makedirs(dirpath_out, exist_ok=True)

    # Generate a range of output rates for each pop.
    rmax = 100
    rates_out_vec = np.geomspace(1, rmax, npts) - 1
    rates_out_mat = np.tile(rates_out_vec, (npops, 1))

    # Convert output rates to NetRegime1DList
    regimes_out = NetRegime1DList.from_regimes_mat(
        pop_names, rates_out_mat
    )

    # Map regimes to inputs
    inputs = [net_ir_mapper.R_to_I(regime) for regime in regimes_out]

    # Extract ou_mean and ou_std matrices from the input list
    ou_mean_mat = np.zeros((npops, npts))
    ou_std_mat = np.zeros((npops, npts))
    for m, inp in enumerate(inputs):
        ou_mean_mat[:, m] = inp.get_pop_inputs_vec('ou_mean')
        ou_std_mat[:, m] = inp.get_pop_inputs_vec('ou_std')

    for n, pop_name in enumerate(pop_names):
        print(f'Plotting I-R mapping for {pop_name}...')

        R = rate_mats[pop_name]
        
        # Slie of the training data
        ou_mean_vec = R.coords['ou_mean'].values
        slicer = net_ir_mapper.pop_IR_mappers[pop_name].slicer
        rr_vec = slicer.get_1d_slice(R, ou_mean_vec)

        # Mask for the training data that was used for fitting
        mask = ((ou_mean_vec >= inp_limits[pop_name][0]) &
                (ou_mean_vec <= inp_limits[pop_name][1]))

        # I-R mapping result
        ou_mean_vec_hat = ou_mean_mat[n, :]
        ou_std_vec_hat = ou_std_mat[n, :]
        rr_vec_hat = rates_out_vec

        ou_mean_vec = ou_mean_vec * 100
        ou_mean_vec_hat = ou_mean_vec_hat * 100
        ou_std_vec_hat = ou_std_vec_hat * 100

        plt.figure(111)
        plt.clf()

        plt.plot(ou_mean_vec_hat, rr_vec_hat, 'b-')
        plt.plot(ou_mean_vec[~mask], rr_vec[~mask], 'kx')
        plt.plot(ou_mean_vec[mask], rr_vec[mask], 'k.', markersize=6)
        plt.title(pop_name)
        plt.xlabel('OU mean * 100')
        plt.ylabel('Rate')

        plt.xlim(0, inp_vis_max)
        plt.ylim(0, r_vis_max)

        plt.show()
        plt.draw()

        fpath_fig = dirpath_out / f'{n}_{pop_name}.png'
        plt.savefig(fpath_fig)
