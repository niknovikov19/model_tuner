import os
import pickle
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.opt.inputs import PopInputOU, NetInputOU
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import (
    PopIRMapper1DSlice, NetIRMapper1DSlice
)
from model_tuner.opt.map_funcs import MapFuncType, MapFitParams

from model_tuner.main import IRMapConfigRateFrom2DRateCVMats

from model_tuner.opt.slicers import LinearSlicer

""" # Needed for unpickling files that were created with the old folder structure
from model_tuner.data_proc import data_types, proc_params
sys.modules['data_types'] = data_types
sys.modules['proc_params'] = proc_params """


def init_params() -> IRMapConfigRateFrom2DRateCVMats:

    par = IRMapConfigRateFrom2DRateCVMats()

    par.fpath_mats = (
        r'D:\WORK\Salvador\repo\model_tuner\test_data\a1_ou_unconn'
        r'\scott_2025_02_28\OUmapping_0228.pkl'
    )

    par.batch_param_names = ('ou_mean', 'ou_std')
    par.batch_param_main = 'ou_mean'

    par.slice_method = 'batch_param_ratio'
    par.batch_param_ratio = 0.4

    # I-R mapping type and hyperparameters
    par.map_type = MapFuncType.RICHARDS_1D
    par.map_params = {
        'x_limits': (0, np.inf),
        'y_limits': (0, np.inf)
    }

    # Fitting bounds for I-R mapping parameters
    par.fit_param_bounds = {
        'q': (1, 30)  # asymmetry coefficient of RICHARDS_1D mapping
    }

    # Limits for the input values used for fitting
    par.inp_limits = {}
    inp_max = {
        'NGF': 3,
        'SOM': 2,
        'PV': 3,
        'VIP': 2,
        'IT23': 2.5,
        'ITP4': 1.5,
        'ITS4': 1,
        'IT5A': 2,
        'IT5B': 1.75,
        'CT56': 1.75,
        'PT5B': 3.5,
        'thal': 3
    }
    inp_max = {pop: val / 100 for pop, val in inp_max.items()}
    layers_2_6 = ['2', '3', '4', '5A', '5B', '6']
    layers_1_6 = ['1'] + layers_2_6
    par.inp_limits |= {f'NGF{layer}': (0, inp_max['NGF']) for layer in layers_1_6}
    par.inp_limits |= {f'SOM{layer}': (0, inp_max['SOM']) for layer in layers_2_6}
    par.inp_limits |= {f'PV{layer}': (0, inp_max['PV']) for layer in layers_2_6}
    par.inp_limits |= {f'VIP{layer}': (0, inp_max['VIP']) for layer in layers_2_6}
    par.inp_limits |= {f'IT{layer}': (0, inp_max['IT23']) for layer in ['2', '3']}
    par.inp_limits |= {'ITP4': (0, inp_max['ITP4'])}
    par.inp_limits |= {'ITS4': (0, inp_max['ITS4'])}
    par.inp_limits |= {'IT5A': (0, inp_max['IT5A'])}
    par.inp_limits |= {'IT5B': (0, inp_max['IT5B'])}
    par.inp_limits |= {f'CT{layer}': (0, inp_max['CT56']) for layer in ['5A', '5B', '6']}
    thal_pops = {'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM'}
    par.inp_limits |= {pop: (0, inp_max['thal']) for pop in thal_pops}

    # Populations used for fitting
    par.pop_names = list(par.inp_limits.keys())
                    
    # Parameters of the formula that determines the fitting weights
    par.use_fit_weights = True
    par.fit_weight_pow = 0.5
    par.fit_weight_limits = (0.1, 10)

    # Parameters of the fitting algorithm
    par.map_fit_params = MapFitParams(
        #return_first_guess=True
    )

    return par


def plot_ir_mapping(
        net_ir_mapper: NetIRMapper1DSlice,
        rate_mats: Dict[str, xr.DataArray],
        slicer: LinearSlicer,
        inp_limits: Dict[str, Tuple[float, float]],
        rvis_max: float = 100,
        npts: int = 100
        ) -> None:
    
    pop_names = net_ir_mapper.pop_names
    npops = len(pop_names)

    # Generate a range of output rates
    rates_out_vec = np.geomspace(1, rvis_max, npts) - 1

    rates_out_mat = np.tile(rates_out_vec, (4, 1))

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
        R = rate_mats[pop_name]

        # Training data
        ou_mean_vec = R.coords['ou_mean'].values
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
        plt.ion()

        plt.subplot(1, 2, 1)
        plt.plot(ou_mean_vec_hat, rr_vec_hat, 'b-')
        plt.plot(ou_mean_vec[~mask], rr_vec[~mask], 'kx')
        plt.plot(ou_mean_vec[mask], rr_vec[mask], 'k.', markersize=6)
        #plt.xlim(ou_mean_vec.min(), ou_mean_vec.max())
        plt.xlim(ou_mean_vec_hat.min(), ou_mean_vec_hat.max())
        plt.ylim(0, rvis_max)
        plt.title(pop_name)
        plt.xlabel('OU mean * 100')
        plt.ylabel('Rate')

        plt.subplot(1, 2, 2)
        plt.plot(ou_mean_vec_hat, ou_std_vec_hat, '.-')
        plt.title(pop_name)
        plt.xlabel('OU mean * 100')
        plt.ylabel('OU std * 100')

        plt.show()
        plt.draw()


# Initialize parameters
par = init_params()

# Load rate and CV matrices
with open(par.fpath_mats, 'rb') as file:
    mats = pickle.load(file)

#pop_names = par.pop_names
pop_names = ['PV3']

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

def weight_func(x: np.ndarray, par_) -> np.ndarray:
    if not par_.use_fit_weights: return None
    return np.clip(x ** par_.fit_weight_pow, *par_.fit_weight_limits)

# Object that takes 1-d slices of the firing rate matrices
# Slice: ou_std = ou_mean * par.batch_param_ratio
slicer = LinearSlicer(
    coord_names=['ou_std', 'ou_mean'],
    coord_main='ou_mean',
    coord_coeffs={'ou_std': (par.batch_param_ratio, 0)}
)

# Network I-R mapper
net_ir_mapper = NetIRMapper1DSlice()

for pop_name in pop_names:
    # Firing rate matrix of a population (ou_std x ou_mean)
    R_ = R[pop_name]

    # Select submatrix of R_ with the main coordinate within the range
    ou_mean_min, ou_mean_max = par.inp_limits[pop_name]
    R_ = R_.sel(ou_mean=slice(ou_mean_min, ou_mean_max))

    # Train pop. I-R mapper on a slice of the firing rate matrix
    pop_ir_mapper = PopIRMapper1DSlice(
        slicer=slicer,
        map_type=par.map_type,
        map_params=par.map_params
    )
    pop_ir_mapper.fit_from_data(
        X=R_,
        fit_params=par.map_fit_params,
        weight_func=lambda x: weight_func(x, par),
        bounds=par.fit_param_bounds
    )

    # Add population I-R mapper to the network I-R mapper
    net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)

# Plot I-R mapping
plot_ir_mapping(net_ir_mapper, R, slicer, par.inp_limits,
                rvis_max=25, npts=250)

input('Press any key to continue...')
