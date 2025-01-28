"""
implementation of batchtk sim manager for hpc
make sure to use the WIP batchtk branch for now as it implements the SSH based dispatcher
"""
from model_tuner.sim_manager import SimManager, SimStatus
from model_tuner.ssh import SSHParams, SSHFSCustom, SSHConnCustom
from enum import Enum, auto
from typing import Dict, List
from batchtk.runtk.dispatchers import SSHDispatcher
from batchtk.runtk.submits import Submit, Template
from batchtk.runtk import STATUS, FILE_HANDLES


# edit this script template per your project specs (i.e.
# smp w/ number of cores, vmem with required mem, h_rt with walltime
# don't change any of the lines with {output_path} or {label}
script_template = \
    """\
#!/bin/bash
#$ -N job{label}
#$ -q cpu.q
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=00:30:00
#$ -o {output_path}/{label}.run
cd {project_path}
{env}
export OUTFILE="{output_path}/{label}.out"
export SGLFILE="{output_path}/{label}.sgl"
export JOBID=$JOB_ID
touch $OUTFILE
echo hello from process $JOBID >> $OUTFILE
echo hello from directory $(pwd) >> $OUTFILE
touch $SGLFILE
"""
submit_template = Template(template="source ~/.bash_profile;source ~/.bashrc;/ddn/age/bin/lx-amd64/qsub {output_path}/{label}.sh",
                           key_args={'output_path', 'label'})
script_template = Template(template=script_template,
                           key_args={'label', 'project_path', 'output_path', 'env', 'command'})
class SGESSHSubmit(Submit):
    def __init__(self, **kwargs):
        super().__init__(
            submit_template = submit_template,
            script_template = script_template,
            handles = FILE_HANDLES,
        )

ssh_par_lethe = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)
ssh_par_grid = SSHParams(
    host='grid',
    user='niknovikov19',
    fpath_private_key=r'C:\Users\aleks\.ssh\id_ed25519_grid'
)

dispatcher_kwargs = {
    'submit': SGESSHSubmit(),
    'remote_dir': '/ddn/jchen/new_project', # the directory where the python script exists on the remote machine
    'remote_out': '/ddn/jchen/new_project_out', # the directory where generated files exist
    'connection': SSHConnCustom([ssh_par_lethe, ssh_par_grid]), #note that just providing a connection instance will create a filesystem object as well
    'fs': SSHFSCustom(ssh_par_lethe) # through the use of connection.sftp(), and may be preferred. however, providing a 'fs' argument will
} # over-ride the .sftp() call, and instead use the provided filesystem object provided it follows FSProtocol (see modification to ssh_fs_custom

class BatchTKSimManager(SimManager):
    def __init__(self, dispatcher_kwargs: Dict):
        super().__init__()
        self.dispatcher_kwargs = dispatcher_kwargs
        self.dispatchers = {}
    def update_status(self) -> None:
        """
    class SimStatus(Enum):
    NEED_PUSH = auto()
    WAITING = auto()
    DONE = auto()
    ERROR = auto()

        """
        for label, sim in self.sims.items():
            dispatcher = self.dispatchers[label]
            dispatcher.open_connections()
            status = dispatcher.check_status().status
            dispatcher.close_connections()
            if status == STATUS.NOTFOUND:
                sim.status = SimStatus.NEED_PUSH # somehow the file for submission was not found on the remote server
            if status == STATUS.PENDING:
                sim.status = SimStatus.WAITING
            if status == STATUS.RUNNING:
                sim.status = SimStatus.WAITING
            if status == STATUS.COMPLETED:
                sim.status = SimStatus.DONE

    def _ready_for_request(self) -> bool:
        return True #to be implemented when decide on concurrency limiter

    def _push_requests(self, labels: List[str]) -> None:
        for label in labels:
            dispatcher = SSHDispatcher(**self.dispatcher_kwargs, label=label)
            dispatcher.update_env(self.sims[label].params)
            dispatcher.create_job()
            dispatcher.submit_job()
            dispatcher.close_connections()
            self.dispatchers[label] = dispatcher



