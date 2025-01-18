"""
implementation of batchtk sim manager for hpc
make sure to use the WIP batchtk branch for now as it implements the SSH based dispatcher
"""
from sim_manager import SimManager, SimStatus
from enum import Enum, auto
from typing import Dict, List
from batchtk.sshtk.dispatchers import SSHDispatcher
from batchtk.runtk.submits import Submit, FILE_HANDLES, Template
from batchtk.runtk import STATUS
from fabric import Connection, Config

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

submit = SGESSHSubmit()
"""
    def __init__(self, submit=None, host=None, remote_dir=None, fs=None,
                 remote_out='.', connection=None, config_path='~/.ssh/config',
                 fabric_config=None, env=None, label=None, **kwargs):
"""

class BatchTKSimManager(SimManager):
    def __init__(self):
        super().__init__()
        self.dispatcher_kwargs = {
            'submit': SSHSubmitSFS(),
            'remote_dir': '/ddn/jchen/project', # the directory where the python script exists on the remote machine
            'remote_out': '/ddn/jchen/project/batch_out', # the directory where generated files exist
            'host': 'grid0',
            'connection': Connection('grid0', config=Config(user_ssh_path='~/.ssh/config')),
        }
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



