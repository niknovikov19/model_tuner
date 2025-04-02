import importlib.util
from pathlib import Path
import sys

from netpyne.batchtools.search import search


def _load_module(fpath_mod):
    mod_spec = importlib.util.spec_from_file_location(
        'module.name', fpath_mod)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules['module.name'] = mod
    mod_spec.loader.exec_module(mod)
    return mod


exp_name = 'batch_test_1'

# Import experiment-specific batch_params.py and get batch params
dirpath_self = Path(__file__).resolve().parent
dirpath_exp = dirpath_self / 'exp_configs' / exp_name
fpath_batch_params = dirpath_exp / 'batch_params.py'
batch_params_mod = _load_module(fpath_batch_params)
params = batch_params_mod.get_batch_params()

sge_config = {
    'queue': 'cpu.q',
    'cores': 30,
    'vmem': '128G',
    'realtime': '1:00:00',
    'command': (
        'conda activate netpyne_batch \n'
        'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \n'
        'cd .. \n'
        'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi run_exp_batch.py')
    }

search(job_type = 'sge',
       comm_type = 'socket',
       label = exp_name,
       params = params,
       output_path = f'exp_results/{exp_name}',
       checkpoint_path = 'exp_logs/ray',
       run_config = sge_config,
       num_samples = 1,
       max_concurrent = 13)
