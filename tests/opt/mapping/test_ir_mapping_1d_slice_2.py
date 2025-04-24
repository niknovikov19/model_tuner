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

from model_tuner.main import IRMapConfigRateFrom2DBatchResLists

from model_tuner.utils import save_yaml, load_yaml, compare_yaml, yaml_diff


r_vis_max = 30
inp_vis_max = 3

dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\test_ir_mapping\test_ir_mapping_list'
    r'\PYR'
)

# Load I-R mapper config from YAML
fpath_yaml = dirpath_base / 'ir_map_params.yaml'
ir_params = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom2DBatchResLists)

# Output folder for plots
dirpath_figs = dirpath_base / 'plots'

# Create I-R mapper
ir_mapper = ir_params.init_ir_mapper(dirpath_figs)

""" # Define target regime
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
inp_target = net_ir_mapper.R_to_I(regime_target) """

#input('Press any key to continue...')
