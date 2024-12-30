from dataclasses import dataclass, is_dataclass, fields
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union, Literal, Any, ClassVar

from netpyne.batchtools.search import search

sys.path.append('/ddn/niknovikov19/repo/model_tuner')
from sim_manager_hpc_batch import SimBatchPaths


# Base folder
dirpath_base = sys.argv[1]

# Collection of hpc paths used by SimManagerHPCBatch
paths = SimBatchPaths.create_default(dirpath_base)

# Read sim labels from the requests json file
with open(paths.requests_file, 'r') as fid:
    requests = json.load(fid)
sim_labels = list(requests.keys())

batch_params = {
    'sim_manager.sim_label': sim_labels,
    'sim_manager.requests_file': [paths.requests_file],
}

print('Batch params:', flush=True)
print(batch_params, flush=True)

sge_config = {
    'queue': 'cpu.q',
    'cores': 1,
    'vmem': '16G',
    'realtime': '0:10:00',
    'command': (
        'conda activate netpyne_batch \n'
        'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \n'
        f'cd {dirpath_base} \n'
        'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi hpc_job_script.py'
    )
}

search(job_type = 'sge',
       comm_type = 'socket',
       label = 'sim_manager_batch',
       params = batch_params,
       output_path = paths.results_dir,
       checkpoint_path = paths.batchtools_dir,
       run_config = sge_config,
       num_samples = 1,
       max_concurrent = 13)
