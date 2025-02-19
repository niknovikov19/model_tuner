import json

from netpyne.batchtools.search import search


# Get cmdline arguments
cmd_args = []

# Batch name
exp_name = cmd_args['batch_name']

# Path to the job script
fpath_job_script = cmd_args['fpath_job_script']

# Get simulation labels that identify combinations of external input params
fpath_sims_info = cmd_args['fpath_sims_info']
with open(fpath_sims_info, 'rb') as fid:
    sims_info = json.load(fid)
sim_labels = list(sims_info.keys())

# Batch params
const_par = {'fpath_sims_info': fpath_sims_info}
batch_params = [{'sim_label': label, 'const_par': const_par}
                for label in sim_labels]

sge_config = {
    'queue': 'cpu.q',
    'cores': 30,
    'vmem': '128G',
    'realtime': '3:30:00',
    'command': ('conda activate netpyne_batch \n'
                'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \n'
                'cd .. \n'
                f'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi {fpath_job_script}')
    }

search(job_type = 'sge',
       comm_type = 'socket',
       label = exp_name,
       params = batch_params,
       output_path = f'../data/{exp_name}',
       checkpoint_path = '../ray',
       run_config = sge_config,
       num_samples = 1,
       max_concurrent = 25)
