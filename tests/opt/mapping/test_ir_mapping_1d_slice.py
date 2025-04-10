import json
import os
from pathlib import Path
import pickle
from pprint import pprint
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from model_tuner.opt.inputs import PopInputOU, NetInputOU
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import (
    PopIRMapper1DSlice, NetIRMapper1DSlice
)
from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.opt.slicers import LinearSlicer

from model_tuner.main import IRMapConfigRateFrom2DRateCVMats

from model_tuner.utils import save_yaml, load_yaml, compare_yaml, yaml_diff

""" # Needed for unpickling files that were created with the old folder structure
from model_tuner.data_proc import data_types, proc_params
sys.modules['data_types'] = data_types
sys.modules['proc_params'] = proc_params """


def init_params() -> IRMapConfigRateFrom2DRateCVMats:

    par = IRMapConfigRateFrom2DRateCVMats()

    par.fpath_mats = (
        r'D:\WORK\Salvador\repo\model_tuner\test_data\a1_ou_unconn'
        #r'\scott_2025_03_13\OUmapping_master_compat.pkl'
        r'\scott_2025_03_26\OUmapping_v45_batch21.pkl'
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
        'q': (1, 30),    # asymmetry coefficient of RICHARDS_1D mapping
        'a': (0, np.inf)
    }

    # Limits for the input values used for fitting
    par.inp_limits = {}
    inp_max = {
        'NGF': 1,
        'SOM': 0.6,
        'PV': 2.2,
        'VIP': 1,
        'IT236': 1.8,
        'ITP4': 1.5,
        'ITS4': 0.7,
        'IT5A': 1.5,
        'IT5B': 1.3,
        'CT56': 1.7,
        'PT5B': 3,
        'TC': 0.7,
        'IRE': 0.4,
        'TI': 0.5
    }
    inp_max = {pop: val / 100 for pop, val in inp_max.items()}
    layers_2_6 = ['2', '3', '4', '5A', '5B', '6']
    layers_1_6 = ['1'] + layers_2_6
    par.inp_limits |= {f'NGF{layer}': (0, inp_max['NGF']) for layer in layers_1_6}
    par.inp_limits |= {f'SOM{layer}': (0, inp_max['SOM']) for layer in layers_2_6}
    par.inp_limits |= {f'PV{layer}': (0, inp_max['PV']) for layer in layers_2_6}
    par.inp_limits |= {f'VIP{layer}': (0, inp_max['VIP']) for layer in layers_2_6}
    par.inp_limits |= {f'IT{layer}': (0, inp_max['IT236']) for layer in ['2', '3', '6']}
    par.inp_limits |= {'ITP4': (0, inp_max['ITP4'])}
    par.inp_limits |= {'ITS4': (0, inp_max['ITS4'])}
    par.inp_limits |= {'IT5A': (0, inp_max['IT5A'])}
    par.inp_limits |= {'IT5B': (0, inp_max['IT5B'])}
    par.inp_limits |= {'PT5B': (0, inp_max['PT5B'])}
    par.inp_limits |= {f'CT{layer}': (0, inp_max['CT56']) for layer in ['5A', '5B', '6']}
    par.inp_limits |= {pop: (0, inp_max['TC']) for pop in ['TC', 'TCM', 'HTC']}
    par.inp_limits |= {pop: (0.0002, inp_max['IRE']) for pop in ['IRE', 'IREM']}
    par.inp_limits |= {pop: (0, inp_max['TI']) for pop in ['TI', 'TIM']}
    
    # Disable all limits
    par.inp_limits = {pop: (0, np.inf) for pop in par.inp_limits.keys()}

    # Populations used for fitting
    par.pop_names = None  #list(par.inp_limits.keys())
                    
    # Parameters of the formula that determines the fitting weights
    par.use_fit_weights = True
    par.fit_weight_pow = 0.6
    par.fit_weight_limits = (0.1, 20)

    # Parameters of the fitting algorithm
    par.map_fit_params = MapFitParams(
        #return_first_guess=True,
        #verbose = True,
        xtol=None,
        ftol=1e-4,
        max_nfev=2000
    )

    return par

def plot_ir_mapping(
        net_ir_mapper: NetIRMapper1DSlice,
        rate_mats: Dict[str, xr.DataArray],
        slicer: LinearSlicer,
        inp_limits: Dict[str, Tuple[float, float]],
        r_vis_max: float = 100,
        inp_vis_max: float = 5,
        npts: int = 100,
        dirpath_plots: Path | str | None = None,
        need_save: bool = True,
        regime_target: NetRegime1D | None = None,
        inp_target: NetInputOU | None = None,
        pop_names_vis: List[str] | None = None
        ) -> None:
    
    pop_names = net_ir_mapper.pop_names
    npops = len(pop_names)

    # Generate a range of output rates
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

        #plt.subplot(1, 2, 1)
        plt.plot(ou_mean_vec_hat, rr_vec_hat, 'b-')
        plt.plot(ou_mean_vec[~mask], rr_vec[~mask], 'kx')
        plt.plot(ou_mean_vec[mask], rr_vec[mask], 'k.', markersize=6)
        plt.plot(inp_target.pop_inputs[pop_name].vars['ou_mean'] * 100,
                 regime_target.get_pop_regime_val(pop_name), 'r.', markersize=10)
        #plt.xlim(ou_mean_vec.min(), ou_mean_vec.max())
        #plt.xlim(ou_mean_vec_hat.min(), ou_mean_vec_hat.max())
        #plt.xlim(0, np.nanmax(ou_mean_vec))
        #plt.ylim(0, np.nanmax(rr_vec))
        plt.title(pop_name)
        plt.xlabel('OU mean * 100')
        plt.ylabel('Rate')

        plt.xlim(0, inp_vis_max)
        plt.ylim(0, r_vis_max)

        """ plt.subplot(1, 2, 2)
        plt.plot(ou_mean_vec_hat, ou_std_vec_hat, '.-')
        plt.title(pop_name)
        plt.xlabel('OU mean * 100')
        plt.ylabel('OU std * 100') """

        plt.show()
        plt.draw()

        if need_save:
            fpath_fig = dirpath_plots / f'{n}_{pop_name}.png'
            plt.savefig(fpath_fig)

def define_target_rates(pop_names) -> Dict[str, float]:    
    cell_rates = {
        'NGF': 20,
        'SOM': 5,
        'PV': 10,
        'VIP': 15,
        'IT': 2,
        'ITP': 2,
        'ITS': 15,
        'CT': 3,
        'PT': 4
    }
    layer_rate_deltas = {
        '1': 0.1,
        '2': 0.2,
        '3': 0.3,
        '4': 0.4,
        '5A': 0.5,
        '5B': 0.55,
        '6': 0.6
    }
    pop_rates = {}
    for cell_name, cell_r in cell_rates.items():
        for layer, layer_dr in layer_rate_deltas.items():
            pop_name = f'{cell_name}{layer}'
            if pop_name in pop_names:
                pop_rates[pop_name] = cell_r + layer_dr
    thal_pops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
    for n, pop_name in enumerate(thal_pops):
        if pop_name in pop_names:
            pop_rates[pop_name] = 10 + 0.1 * n
    return pop_rates


# Initialize parameters
par = init_params()

# Load rate and CV matrices
with open(par.fpath_mats, 'rb') as file:
    mats = pickle.load(file)
par.pop_names = list(mats['rate'].keys())

pop_names = par.pop_names
#pop_names = ['SOM3']

r_vis_max = 30
inp_vis_max = 3

dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\test_ir_mapping'
    r'\scott_2025_03_26'
    fr'\test_ir_mapping_1d_slice_irreg_rmax={r_vis_max}_imax={inp_vis_max}'
)
os.makedirs(dirpath_base, exist_ok=True)

need_recalc = 1
need_save_yaml = 1
replace_old_yaml = 1
need_save_csv = 1
need_save_json = 1
need_plot = 1
need_save_plot = 1
need_save_target = 1

fpath_ir_mapper = dirpath_base / 'ir_mapper.pkl'

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
    """Calculate weights for fitting based on the firing rates. """
    if not par_.use_fit_weights: return None
    x = np.clip(x, 0, np.inf)
    return np.clip(x ** par_.fit_weight_pow, *par_.fit_weight_limits)

# Object that takes 1-d slices of the firing rate matrices
# Slice: ou_std = ou_mean * par.batch_param_ratio
slicer = LinearSlicer(
    coord_names=['ou_std', 'ou_mean'],
    coord_main='ou_mean',
    coord_coeffs={'ou_std': (par.batch_param_ratio, 0)}
)

# Fit I-R mapping
if need_recalc or not fpath_ir_mapper.exists():

    # Network I-R mapper
    net_ir_mapper = NetIRMapper1DSlice()

    for pop_name in pop_names:
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
            weight_func=lambda x: weight_func(x, par),
            bounds=par.fit_param_bounds
        )

        # Add population I-R mapper to the network I-R mapper
        net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)

    # Save the network I-R mapper
    with open(fpath_ir_mapper, 'wb') as file:
        pickle.dump(net_ir_mapper, file)

else:
    # Load I-R mapper
    with open(fpath_ir_mapper, 'rb') as file:
        net_ir_mapper = pickle.load(file)

# Define target regime
target_rates = define_target_rates(pop_names)
regime_target = NetRegime1D.from_dict(target_rates)

# Save target regime to csv
if need_save_target:
    data = {
        'pop_name': list(target_rates.keys()),
        'target_rate': list(target_rates.values())
    }
    df = pd.DataFrame(data)
    csv_path = dirpath_base / 'target_rates.csv'
    df.to_csv(csv_path, index=False)

# Apply I-R mapping to the target regime
inp_target = net_ir_mapper.R_to_I(regime_target)

# Save I-R mapper config to YAML
if need_save_yaml:
    fpath_yaml = dirpath_base / 'ir_mapper_config.yaml'
    if not os.path.exists(fpath_yaml) or replace_old_yaml:
        save_yaml(par, fpath_yaml)
        print(f'Config saved to {fpath_yaml}')
    # Load I-R mapper config from YAML
    par_loaded = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom2DRateCVMats)
    # Compare original and loaded I-R mapper configs
    if compare_yaml(par, par_loaded):
        print('Loaded config is the same as the original one')
    else:
        pprint(yaml_diff(par, par_loaded))
        raise ValueError('Error: loaded config is different from the original one')

# Create input and output values to CSV
if need_save_csv:
    data = {
        'pop_name': [],
        'rate': [],
        'ou_mean': [],
        'ou_std': []
    }
    for pop_name in pop_names:
        data['pop_name'].append(pop_name)
        data['rate'].append(regime_target.get_pop_regime_val(pop_name))
        data['ou_mean'].append(inp_target.pop_inputs[pop_name].vars['ou_mean'])
        data['ou_std'].append(inp_target.pop_inputs[pop_name].vars['ou_std'])
    df = pd.DataFrame(data)
    csv_path = dirpath_base / 'regime_target.csv'
    df.to_csv(csv_path, index=False)

if need_save_json:
    data = {'ou_inputs': {}}
    for pop_name in pop_names:
        data['ou_inputs'][pop_name] = {
            'ou_mean': inp_target.pop_inputs[pop_name].vars['ou_mean'],
            'ou_std': inp_target.pop_inputs[pop_name].vars['ou_std']
        }
    json_path = dirpath_base / 'ou_inputs.json'
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

# Plot I-R mapping
if need_plot:
    dirpath_plots = dirpath_base / 'plots'
    os.makedirs(dirpath_plots, exist_ok=True)
    plot_ir_mapping(
        net_ir_mapper, R, slicer, par.inp_limits,
        r_vis_max=r_vis_max,
        inp_vis_max=inp_vis_max,
        npts=250,
        dirpath_plots=dirpath_plots, need_save=need_save_plot,
        regime_target=regime_target, inp_target=inp_target
    )

input('Press any key to continue...')
