from pathlib import Path

import pandas as pd

from model_tuner.main import (
    IRMapConfigRateFrom1DBatchResXR,
    plot_ir_mapping_1d_slice
)

from model_tuner.utils import load_yaml


dirpath_base = Path(
    r'D:\WORK\Salvador\repo\model_tuner\test_data\test_ir_mapping'
    r'\test_ir_mapping_1d_slice_5'
)

rmax = None
imax = None

# Load I-R mapper config from YAML
fpath_yaml = dirpath_base / 'ir_map_params.yaml'
ir_params = load_yaml(fpath_yaml, data_class=IRMapConfigRateFrom1DBatchResXR)

# Load target regime
csv_path = dirpath_base / 'regime_target.csv'
if csv_path.exists():
    df = pd.read_csv(csv_path)
    target_rates = dict(zip(df['pop_name'], df['target_rate']))
else:
    target_rates = None

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
    target_rates,
    r_vis_max=rmax,
    inp_vis_max=imax,
    npts=5000,
    rbase=1e-16
)

#input('Press any key to continue...')
