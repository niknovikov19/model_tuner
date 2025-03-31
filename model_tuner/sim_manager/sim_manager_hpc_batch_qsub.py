from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Union, ClassVar, Optional

from ..ssh import SSHClient
from ..utils import CustomEncoder

from .sim_manager import SimManager, SimStatus
from .sim_manager_hpc_batch import SimManagerHPCBatch, SimBatchPaths
import time


def _joinpath_hpc(base, *args):
    return Path(base).joinpath(*args).as_posix()

def _get_parent_dir_hpc(fpath):
    return Path(fpath).parent.as_posix()


@dataclass
class HPCJobSubmitParams:
    num_cores: int = 1
    queue: str = 'cpu.q'
    memory: str | int = '32G'
    time_limit: str | None = '12:00:00'

    def __post_init__(self):
        # Convert memory to string if it is an integer
        if isinstance(self.memory, int):
            self.memory = f'{self.memory}G'


def _create_qsub_bash_header(
        job_name: str,
        job_par: HPCJobSubmitParams,
        fpath_log: str,
        fpath_err: str | None = None
        ) -> str:
    """Create a bash header for the batch script."""
    lines = []
    lines.append('#!/bin/bash')
    lines.append(f'#$ -cwd')  # run in the current working directory
    lines.append(f'#$ -N {job_name}')
    lines.append(f'#$ -q {job_par.queue}')
    lines.append(f'#$ -pe smp {job_par.num_cores}')
    lines.append(f'#$ -l h_vmem={job_par.memory}')
    lines.append('#$ -V')
    if job_par.time_limit:
        lines.append(f'#$ -l h_rt={job_par.time_limit}')
    lines.append(f'#$ -o {fpath_log}')
    if fpath_err:
        lines.append(f'#$ -e {fpath_err}')
    return '\n'.join(lines)


class SimManagerHPCBatchQsub(SimManagerHPCBatch):
    """
    Same as SimManagerHPCBatch, but the batch script is itself submitted via qsub.

    A sh-file with '_submit' postfix is created in the same folder as the batch script.

    """

    BATCH_JOB_NAME: ClassVar[str] = 'BATCH_MAIN'
    
    def __init__(
            self,
            ssh: SSHClient,
            fpath_batch_script: str,
            batch_script_job_params: HPCJobSubmitParams,
            batch_paths: SimBatchPaths,
            conda_env: Optional[str] = None,
            res_filename_templ = '{sim_label}_data.pkl'
            ):
        #self.BATCH_JOB_NAME = 'BATCHTOOLS1'
        self._batch_script_job_params = batch_script_job_params
        super().__init__(ssh, fpath_batch_script, batch_paths, conda_env, res_filename_templ)
    
    def _update_batch_script_status(self) -> None:
        cmd = f'bash -l -c "qstat -f -u \'*\' | grep \\\"{self.BATCH_JOB_NAME}\\\""' 
        self._is_batch_script_running = (
            self._ssh.conn.run(cmd, hide=True, warn=True).ok
        )
        # Test the output of qstat
        #qstat_res = self._ssh.conn.run('bash -l -c \'qstat -f -u "*"\'', hide=True, warn=True)
        #print(f'\nQSTAT\n{qstat_res.stdout.strip()}')
    
    def _run_hpc_script(self):
        raise ValueError('Directly running scripts on HPC is prohibited')

    def _submit_hpc_job(self,
                        fpath_script: str,    # python script to run via qsub
                        job_name: str,
                        job_params: HPCJobSubmitParams,
                        fpath_log: str,
                        fpath_err: str | None = None,
                        cmd_args: str | List[str] = field(default_factory=list)
                        ) -> None:
        """Create sh-file that runs the script and submit it via qsub. """

        # Create the header of the sh-file that runs the script
        sh_header = _create_qsub_bash_header(
            job_name, job_params, fpath_log, fpath_err
        )

        # Create string of space-separated quoted arguments
        if isinstance(cmd_args, str):
            cmd_args = [cmd_args]
        cmd_args = ' '.join([f'"{arg}"' for arg in cmd_args])  # add quotes
        
        # Get the directory and the name of the script
        dirpath_script = Path(fpath_script).parent.as_posix()
        fname_script = str(Path(fpath_script).name)

        # Body of the sh-file: set conda env, run python script in background,
        # redirect outputs, and make it survive ssh disconnection
        sh_body = []
        sh_body.append('source ~/.bashrc')
        sh_body.append(f'conda activate {self._conda_env}')
        sh_body.append(f'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH')
        sh_body.append(f'cd {dirpath_script}')
        sh_body.append(f'python {fname_script} {cmd_args}')
        sh_body = '\n'.join(sh_body)

        # Write the sh-file to the HPC
        fpath_sh = fpath_script.replace('.py', '_submit.sh')
        with self._ssh.fs.open(fpath_sh, 'w') as fid:
            fid.write(sh_header + '\n\n' + sh_body)

        # Submit the script
        cmd = f"""
            bash -l -c '(
                cd {dirpath_script}
                qsub {fpath_sh}
            )'
        """
        #print(f'COMMAND: \n {cmd}')
        self._ssh.conn.run(cmd, hide=True)
    
    def _push_requests(self, labels: list[str]) -> None:
        if not labels:
            return
        
        # Check that the previous bath is finished, before pushing a new one
        if self._is_batch_script_running:
            raise ValueError('Cannot push, previous batch is still in progress')
            
        # Store params of simulations into a json file on HPC
        fpath_reqs_json = self._paths.requests_file
        self._sim_requests_to_hpc_json(fpath_reqs_json, labels)
        
        # Run the batch script on HPC (via qsub),
        # pass the path to the json file as a command line argument
        self._submit_hpc_job(
            fpath_script=self._fpath_batch_script,
            job_name=self.BATCH_JOB_NAME,
            job_params=self._batch_script_job_params,
            fpath_log=self._paths.log_file,
            fpath_err=self._paths.log_file.replace('.out', '.err'),
            cmd_args=[self._paths.base_dir]
        )

        # Pause for a while to let the batch job be submitted
        time.sleep(5)
        
        # Update statuses of the pushed simulations
        for label in labels:
            self.sims[label].status = SimStatus.WAITING
        
        self._is_batch_script_running = True
    