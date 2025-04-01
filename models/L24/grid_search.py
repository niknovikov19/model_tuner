from pathlib import Path

from netpyne.batchtools.search import search

from load_module import load_module


# Folder name for experiment configs (relative to this script)
DIRNAME_EXP_CONFIGS = 'exp_configs'

exp_name = 'batch_test_1'

# Import experiment-specific batch_params.py and get batch params
fpath_batch_params = (
    Path(__file__).resolve().parent /
    DIRNAME_EXP_CONFIGS / exp_name /
    'batch_params.py'
)
batch_params_mod = load_module(fpath_batch_params)
params = batch_params_mod.get_batch_params()

sge_config = {
    'queue': 'cpu.q',
    'cores': 30,
    'vmem': '128',
    'realtime': '1:00:00',
    'command': (
        'conda activate netpyne_batch \n'
        'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \n'
        'cd .. \n'
        'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi run_exp.py --batch')
}

search(
    job_type='sge',
    comm_type='socket',
    label=exp_name,
    params=params,
    output_path=f'exp_results/{exp_name}',
    checkpoint_path='exp_logs/ray',
    run_config=sge_config,
    num_samples=1,
    max_concurrent=13
)
