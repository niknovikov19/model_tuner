from dataclasses import dataclass, is_dataclass, asdict
import json
from pathlib import Path
from typing import Dict, List, Union, ClassVar, Optional

from json_encoders import CustomEncoder
from sim_manager import SimManager, SimStatus
from ssh_client import SSHClient


def _joinpath_hpc(base, *args):
    return Path(base).joinpath(*args).as_posix()

def _get_parent_dir_hpc(fpath):
    return Path(fpath).parent.as_posix()


@dataclass
class SimBatchPaths:
    base_dir: str
    requests_file: str
    results_dir: str
    log_file: str
    batchtools_dir: str
    
    FILE_FIELDS: ClassVar[List[str]] = [
        'requests_file',
        'log_file'
    ]
    FOLDER_FIELDS: ClassVar[List[str]] = [
        'base_dir',
        'results_dir', 
        'batchtools_dir'
    ]
    
    def get_used_folders(self) -> List[str]:
        folder_paths = [getattr(self, f) for f in self.FOLDER_FIELDS]
        file_parent_dirs = [
            _get_parent_dir_hpc(getattr(self, f)) for f in self.FILE_FIELDS
        ]
        return folder_paths + file_parent_dirs
    
    def get_all_files(self) -> List[str]:
        return [getattr(self, field) for field in self.FILE_FIELDS]            
    
    @classmethod
    def create_default(cls, dirpath_base: str) -> 'SimBatchPaths':
        paths_rel = {
            'requests_file': 'requests/requests.json',
            'results_dir': 'results',
            'log_file': 'log/batch_script_log.out',
            'batchtools_dir': 'batchtools'
        }
        paths_abs = {key: _joinpath_hpc(dirpath_base, path_rel)
                     for key, path_rel in paths_rel.items()}
        paths_abs['base_dir'] = dirpath_base        
        return cls(**paths_abs)


class SimManagerHPCBatch(SimManager):
    
    def __init__(
            self,
            ssh: SSHClient,
            fpath_batch_script: str,
            batch_paths: SimBatchPaths,
            conda_env: Optional[str] = None
            ):
        super().__init__()
        self._ssh = ssh
        self._fpath_batch_script = fpath_batch_script
        self._paths = batch_paths
        self._conda_env = conda_env or 'base'
        self._is_batch_script_running = False
        self.update_status()
    
    def get_sim_result_path(self, label: str) -> str:
        return (Path(self._paths.results_dir) / f'{label}.pkl').as_posix()
    
    def _update_batch_script_status(self) -> None:
        proc_str = f'python {self._fpath_batch_script}'
        cmd = f'ps aux | grep "{proc_str}" | grep -v grep'
        self._is_batch_script_running = (
            self._ssh.conn.run(cmd, hide=True, warn=True).ok
        )
    
    def _update_sim_status(self, label: str) -> None:
        sim = self.sims[label]
        # Statuses other than WAITING don't require an update
        if sim.status != SimStatus.WAITING:
            return
        
        # If the batch script is still running - consider the sims not ready
        if self._is_batch_script_running:
            return         
        
        # Check whether the result file of this simulation exists
        # (after completion of the batch script)
        fpath_res = self.get_sim_result_path(label)
        if self._ssh.fs.exists(fpath_res):
            sim.status = SimStatus.DONE
        else:
            sim.status = SimStatus.ERROR  # batch finished, but no result       
    
    def update_status(self) -> None:
        """Update statuses of simulation requests. """        
        self._update_batch_script_status()
        for label in self.sims:
            self._update_sim_status(label)
    
    def _ready_for_request(self) -> bool:
        """Check whether the object is ready for add_sim_request(). """
        ready = (
            not self.is_waiting(update=False) and  # no requests being processed
            not self._is_batch_script_running
        )
        return ready
    
    def _sim_requests_to_hpc_json(
            self,
            fpath_json: str,
            labels_used: List[str] = None
            ) -> None:
        if labels_used is None: labels_used = self.sims.keys()
        sim_reqs = {label: self.sims[label].params for label in labels_used}
        with self._ssh.fs.open(fpath_json, 'w') as fid:
            json.dump(sim_reqs, fid, cls=CustomEncoder)
    
    def _run_hpc_script(self,
                        fpath_script: str,
                        fpath_log: str,
                        cmd_args: Union[str, List[str]]
                        ) -> None:
        # Create string of space-separated quoted arguments
        if isinstance(cmd_args, str):
            cmd_args = [cmd_args]
        cmd_args = ' '.join([f'"{arg}"' for arg in cmd_args])  # add quotes
        # Command to run: set conda env, run python script in background,
        # redirect outputs, and make it survive ssh disconnection
        dirpath_script = Path(fpath_script).parent.as_posix()
        cmd = f"""
            bash -l -c '(
                conda activate {self._conda_env}
                cd {dirpath_script}
                nohup python {fpath_script} {cmd_args} > {fpath_log} 2>&1 &
            )'
        """
        #print(f'COMMAND: \n {cmd}')
        # Run the command via ssh
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
        
        # Run batch script on HPC, pass the path to the json file  as command line arguments
        self._run_hpc_script(
            self._fpath_batch_script,
            self._paths.log_file,
            cmd_args=[self._paths.base_dir]
        )
        
        # Update statuses of the pushed simulations
        for label in labels:
            self.sims[label].status = SimStatus.WAITING
        
        self._is_batch_script_running = True
    