from dataclasses import dataclass, is_dataclass, fields
from enum import Enum, auto
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal, Any

from fabric import Connection as FabricConnection
from fs.sshfs import SSHFS

from sim_manager import SimManager, SimStatus
from ssh_client import SSHClient


class SimManagerHPCBatch(SimManager):
    
    HPC_DIR_REQUESTS_DEF = ''
    HPC_DIR_RESULTS_DEF = ''
    HPC_DIR_LOGS_DEF = ''
    
    def __init__(
            self,
            ssh: SSHClient,
            fpath_bacth_script: str,
            hpc_dir_requests: str = None,
            hpc_dir_results: str = None,
            hpc_dir_logs: str = None,
            ):
        super().__init__()
        self._fpath_bacth_script = fpath_bacth_script
        self._hpc_dir_requests = hpc_dir_requests or self.HPC_DIR_REQUESTS_DEF
        self._hpc_dir_results = hpc_dir_results or self.HPC_DIR_RESULTS_DEF
        self._hpc_dir_logs = hpc_dir_logs or self.HPC_DIR_LOGS_DEF
        self._is_req_batch_pushed = False
        self._ssh = ssh
    
    def _ready_for_request(self) -> bool:
        return not self._is_req_batch_pushed
    
    def _gen_sim_requests_json_path(self) -> str:
        return (Path(self._hpc_dir_requests) / 'sim_requests.json').as_posix()
    
    def _gen_sim_result_path(self, label: str) -> str:
        return (Path(self._hpc_dir_results) / f'{label}.pkl').as_posix()
    
    def _gen_log_path(self) -> str:
        return (Path(self._hpc_dir_logs) / 'batch_log.out').as_posix()

    def _update_sim_status(self, label: str) -> None:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        if self.sims[label].status == SimStatus.WAITING:
            # Mark simulation as DONE if its result file appeared on hpc
            fpath_res = self._gen_sim_result_path(label)
            if self._ssh.fs.exists(fpath_res):
                self.sims[label].status = SimStatus.DONE
    
    def _update_all_sim_statuses(self) -> None:
        for label in self.sims:
            self._update_sim_status(label)
    
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
                        fpath_out: str,
                        cmd_args: Union[str, List[str]]
                        ) -> None:
        # Create string of space-separated quoted arguments
        if isinstance(cmd_args, str):
            cmd_args = [cmd_args]
        cmd_args = ' '.join([f'"{arg}"' for arg in cmd_args])  # add quotes
        # Command to run: background, survive ssh disconnection, redirect outputs  
        cmd = f'nohup python {fpath_script} {cmd_args} > {fpath_out} 2>&1 &'
        # Run the command via ssh
        self._ssh.conn.run(cmd, hide=True)
    
    def push_all_requests(self) -> None:
        # Simulations that await pushing
        sims_to_push = {label: sim for label, sim in self.sims.items()
                        if sim.status == SimStatus.NEED_PUSH}
        if len(sims_to_push) != 0 and self._is_req_batch_pushed:
            raise ValueError('Cannot push, previous batch is still processing')
        # Store params of simulations into a json file on HPC
        fpath_reqs_json = self._gen_sim_requests_json_path()
        self._sim_requests_to_hpc_json(fpath_reqs_json, sims_to_push.keys())
        # Run batch script on HPC, pass the path to the json file 
        # and the path to the output folder as command line arguments
        fpath_log = self._gen_log_path()
        self._run_hpc_script(
            self._fpath_bacth_script,
            fpath_log,
            cmd_args=[fpath_reqs_json, self._hpc_dir_results]
        )
        # Update statuses
        for sim in sims_to_push.values():
            sim.status = SimStatus.WAITING
        self._is_req_batch_pushed = True
        

    