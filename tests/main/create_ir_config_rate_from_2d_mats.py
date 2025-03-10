import os
from pprint import pprint

import numpy as np

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.main import IRMapConfigRateFrom2DRateCVMats
from model_tuner.utils import save_yaml, load_yaml, compare_yaml, yaml_diff


par = IRMapConfigRateFrom2DRateCVMats()
par.pop_names = ('L2e', 'L2i', 'L4e', 'L4i')

print(f'Secondary param: {par.batch_param_sec}')
par.batch_param_names = ('ouamp', 'oustd')
par.batch_param_main = 'ouamp'
print(f'Secondary param: {par.batch_param_sec}')

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
    'q': (1, 10)  # asymmetry coefficient of RICHARDS_1D mapping
}

# Limits for the input rates used for fitting
par.inp_limits = {
    'L2e': (0, 250),
    'L2i': (0, 1000),
    'L4e': (0, 250),
    'L4i': (0, 1000)
}

# Parameters of the formula that determines the fitting weights
par.use_fit_weights = True
par.fit_weight_pow = 0.5
par.fit_weight_limits = (0.1, 10)

# Parameters of the fitting algorithm
par.map_fit_params = MapFitParams(
    #return_first_guess=True
)

dirpath_base = (
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main'
    r'\create_ir_map_config_rate_from_2d_mats'
)
os.makedirs(dirpath_base, exist_ok=True)
fpath_yaml = os.path.join(dirpath_base, 'config_new.yaml')
replace_old = False

# Save to YAML
if not os.path.exists(fpath_yaml) or replace_old:
    save_yaml(par, fpath_yaml)
    print(f'Config saved to {fpath_yaml}')

# Load from YAML
par_loaded = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom2DRateCVMats)

# Compare original and loaded configs
if compare_yaml(par, par_loaded):
    print('Loaded config is the same as the original one')
else:
    print('Error: loaded config is different from the original one')
    pprint(yaml_diff(par, par_loaded))
