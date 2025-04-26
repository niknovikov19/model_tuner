from pathlib import Path

import pandas as pd

from model_tuner.main import (
    IRMapConfigRateFrom2DBatchResLists,
    plot_ir_mapping_1d_slice
)

from model_tuner.utils import load_yaml


dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\test_ir_mapping\test_ir_mapping_list'
    r'\ALL'
)

# Load I-R mapper config from YAML
fpath_yaml = dirpath_base / 'ir_map_params.yaml'
ir_params = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom2DBatchResLists)

# Load target regime
csv_path = dirpath_base / 'target_rates.csv'
df = pd.read_csv(csv_path)
target_rates = dict(zip(df['pop_name'], df['target_rate']))

# Create and fit I-R mapper
ir_mapper, rate_mats = ir_params.init_ir_mapper()

# Output folder for plots
dirpath_figs = dirpath_base / 'plots'

# Plot I-R mapping
plot_ir_mapping_1d_slice(
    ir_mapper,
    rate_mats,
    ir_params.inp_limits,
    dirpath_figs,
    target_rates
)

#input('Press any key to continue...')
