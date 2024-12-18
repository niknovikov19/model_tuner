from dataclasses import dataclass, is_dataclass, fields
from enum import Enum, auto
import json
import os
from typing import Dict, List, Tuple, Union, Literal, Any

from fabric import Connection as FabricConnection
from fs.sshfs import SSHFS


class SimManagerHPCBatch(SimManager):
    
    HPC_DIR_REQUESTS_DEF = ''
    HPC_DIR_RESULTS_DEF = ''
    HPC_DIR_LOGS_DEF = ''
    
    def __init__(
            self,
            fpath_bacth_script: str,
            hpc_dir_requests: str = None,
            hpc_dir_results: str = None,
            hpc_dir_logs: str = None,
            ):
        super().init()
        self._fpath_bacth_script = fpath_bacth_script
        self._hpc_dir_requests = hpc_dir_requests or self.HPC_DIR_REQUESTS_DEF
        self._hpc_dir_results = hpc_dir_results or self.HPC_DIR_RESULTS_DEF
        self._hpc_dir_logs = hpc_dir_logs or self.HPC_DIR_LOGS_DEF
        self._is_req_batch_pushed = False
        self._ssh_host = ''
        self._ssh_user = ''
        self._ssh_password = ''
    
    def _ready_for_request(self) -> bool:
        return not self._is_req_batch_pushed
    
    def _gen_sim_requests_json_path(self) -> str:
        return os.path.join(self.hpc_dir_requests, 'sim_requests.json')
    
    def _gen_sim_result_path(self, label: str) -> str:
        return os.path.join(self._hpc_dir_results, f'{label}.pkl')
    
    def _get_sshfs_handler(self) -> SSHFS:
        return SSHFS(f'{self._ssh_username}@{self._ssh_host}',
                     passwd=self._ssh_password)
    
    def _get_ssh_conn_handler(self) -> FabricConnection:
        return FabricConnection(
            host=self._ssh_host, user=self._ssh_username,
            connect_kwargs={'password': self._ssh_password}
        )
    
    def _ssh_file_exists(self, fpath: str) -> bool:
        with self._get_sshfs_handler() as fs:
            return fs.exists(fpath)

    def _update_sim_status(self, label: str) -> None:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        if self.sims[label].status == SimStatus.WAITING:
            # Mark simulation as DONE if its result file appeared on hpc
            fpath_res = self._gen_sim_result_path(label)
            if self._ssh_file_exists(fpath_res):
                self.sims[label].status = SimStatus.DONE    
    
    def _init_sim_request(self, label: str, params: Dict, push_now: bool) -> None:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        if self.sims[label].status != SimStatus.NEED_PUSH:
            raise ValueError(f'Simulation {label} should have NEED_PUSH status')
        if push_now:
            raise ValueError('SimManagerHPCBatch does not support pushing '
                             'of individual simulations, use push_all_requests()')
    
    def _sim_requests_to_json(
            self,
            fpath_json: str,
            labels_used: List[str] = None
            ) -> None:
        if labels_used is None: labels_used = self.sims.keys()
        sim_reqs = {label: self.sims[label].params for label in labels_used}
        with self._get_sshfs_handler() as fs:
            with fs.open(fpath_json, 'w') as fid:
                json.dump(sim_reqs, fid)
    
    def _ssh_run_script(self,
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
        with self._get_ssh_conn_handler() as conn:
            conn.run(cmd, hide=True)
    
    def push_all_requests(self) -> None:
        # Simulations that await pushing
        sims_to_push = {label: sim for label, sim in self.sims.items()
                        if sim.status == SimStatus.NEED_PUSH}
        if len(sims_to_push) != 0 and self._is_req_batch_pushed:
            raise ValueError('Cannot push, previous batch is still processing')
        # Store params of simulations into a json file on HPC
        fpath_reqs_json = self._gen_sim_requests_json_path()
        self._sim_requests_to_json(fpath_reqs_json, sims_to_push.keys())
        # Run batch script on HPC, pass the path to the json file as an argument
        fpath_log = os.path.join(self._hpc_dir_logs, 'batch_log.txt')
        self._ssh_run_script(self._fpath_bacth_script, fpath_log,
                             cmd_args=fpath_reqs_json)
        # Update statuses
        for sim in sims_to_push.values():
            sim.status = SimStatus.WAITING
        self._is_req_batch_pushed = True
        

    