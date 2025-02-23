from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from io import IOBase
import os
from pprint import pprint
from typing import Any, Protocol

import dacite
from deepdiff import DeepDiff
import numpy as np
import yaml

from model_tuner.opt.map_funcs import MapFuncType, MapFitParams
from model_tuner.data_proc import NetSpikesParams, NetRatesParams

from model_tuner.main import IRMapFitParams


par = IRMapFitParams()
par.pop_names = ('L2e', 'L2i', 'L4e', 'L4i')

# Parameters of the batch experiment that probes a range of input values
par.dirpath_batch = (
    r'D:\WORK\Salvador\repo\model_tuner\models\L24\exp_results\rx_batch_unconn_2'
)
par.exp_name = 'rx_batch_unconn_2'
par.batch_param_name = 'ext_inp_rate_common'

# Parameters of the simulation result processing
proc_params = {
    'net_spikes': NetSpikesParams(pop_names=par.pop_names),  # sim_result -> spikes
    'net_rates': NetRatesParams(time_limits=(0.5, None))     # spikes -> rates
}

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
par.fit_weight_pow = 0.5
par.fit_weight_limits = (0.1, 10)

# Parameters of the fitting algorithm
par.map_fit_params = MapFitParams(
    #return_first_guess=True
)

dirpath_base = r'D:\WORK\Salvador\repo\model_tuner\test_data\main\create_ir_map_config'
os.makedirs(dirpath_base, exist_ok=True)
fpath_yaml = os.path.join(dirpath_base, 'config.yaml')
replace_old = False

def _prepare_for_yaml(obj):
    """Prepare an object for saving to YAML. """
    if is_dataclass(obj):
        obj = asdict(obj)  # dataclass -> dict
    if isinstance(obj, tuple):
        return [_prepare_for_yaml(item) for item in obj]  # tuple -> list
    elif isinstance(obj, list):
        return [_prepare_for_yaml(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _prepare_for_yaml(value) for key, value in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value  # Enum -> str
    else:
        return obj

def save_yaml(obj: Any, fpath_yaml: str) -> None:
    """Save an object to YAML. """
    obj = _prepare_for_yaml(obj)
    with open(fpath_yaml, 'w') as fid:
        yaml.dump(obj, fid, default_flow_style=False)

def compare_yaml(obj1: Any, obj2: Any) -> bool:
    """Compare two objects saved to YAML. """
    obj1 = _prepare_for_yaml(obj1)
    obj2 = _prepare_for_yaml(obj2)
    return obj1 == obj2

class DataclassProtocol(Protocol):
    """Protocol for dataclasses. """
    __dataclass_fields__: dict  # all dataclasses have this attribute

def load_yaml(
        fpath_yaml,
        data_class: DataclassProtocol | None = None
        ) -> DataclassProtocol | dict:
    """Load an object from YAML. """
    with open(fpath_yaml, 'r') as fid:
        obj = yaml.safe_load(fid)
    if data_class is not None:
        if not is_dataclass(data_class):
            raise ValueError('data_class argument should be a dataclass')
        obj = dacite.from_dict(
            data_class=data_class,
            data=obj,
            config=dacite.Config(cast=[tuple, Enum])
        )
    return obj

# Save to YAML
if not os.path.exists(fpath_yaml) or replace_old:
    save_yaml(par, fpath_yaml)
    print(f'Config saved to {fpath_yaml}')

# Load from YAML
par_loaded = load_yaml(fpath_yaml, data_class=IRMapFitParams)

# Compare original and loaded configs
if compare_yaml(par, par_loaded):
    print('Loaded config is the same as the original one')
else:
    print('Error: loaded config is different from the original one')
    pprint(DeepDiff(par, par_loaded))
