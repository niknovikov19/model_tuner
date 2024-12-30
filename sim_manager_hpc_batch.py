from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Union, ClassVar, Optional

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
        self._is_req_batch_pushed = False        
    
    def _ready_for_request(self) -> bool:
        return not self._is_req_batch_pushed
    
    def get_sim_result_path(self, label: str) -> str:
        return (Path(self._paths.results_dir) / f'{label}.pkl').as_posix()

    def _update_sim_status(self, label: str) -> None:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        if self.sims[label].status == SimStatus.WAITING:
            # Mark simulation as DONE if its result file appeared on hpc
            fpath_res = self.get_sim_result_path(label)
            if self._ssh.fs.exists(fpath_res):
                self.sims[label].status = SimStatus.DONE

    def _init_sim_request(self, label: str, params: Dict, push_now: bool) -> None:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        if self.sims[label].status != SimStatus.NEED_PUSH:
            raise ValueError(f'Simulation {label} should have NEED_PUSH status')
        if push_now:
            raise ValueError('SimManagerHPCBatch does not support pushing '
                             'of individual simulations, use push_all_requests()')
    
    def _sim_requests_to_hpc_json(
            self,
            fpath_json: str,
            labels_used: List[str] = None
            ) -> None:
        if labels_used is None: labels_used = self.sims.keys()
        sim_reqs = {label: self.sims[label].params for label in labels_used}
        with self._ssh.fs.open(fpath_json, 'w') as fid:
            json.dump(sim_reqs, fid)
    
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
            )&' &
        """
        #print(f'COMMAND: \n {cmd}')
        # Run the command via ssh
        self._ssh.conn.run(cmd, hide=True)
    
    def push_all_requests(self) -> None:
        # Simulations that await pushing
        sims_to_push = {label: sim for label, sim in self.sims.items()
                        if sim.status == SimStatus.NEED_PUSH}
        if len(sims_to_push) != 0 and self._is_req_batch_pushed:
            raise ValueError('Cannot push, previous batch is still processing')
        # Store params of simulations into a json file on HPC
        fpath_reqs_json = self._paths.requests_file
        self._sim_requests_to_hpc_json(fpath_reqs_json, sims_to_push.keys())
        # Run batch script on HPC, pass the path to the json file  as command line arguments
        self._run_hpc_script(
            self._fpath_batch_script,
            self._paths.log_file,
            #cmd_args=[fpath_reqs_json, self._paths.results_dir]
            cmd_args=[self._paths.base_dir]
        )
        # Update statuses
        for sim in sims_to_push.values():
            sim.status = SimStatus.WAITING
        self._is_req_batch_pushed = True
        

    