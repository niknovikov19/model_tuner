from typing import Dict, List, Tuple, Union, Literal, Any

from netpyne.specs import NetParams


class SimManager:    
    sims_info: Dict[str, Any]
    
    def add_sim_info(self, label: str, info: Any):
        self.sims_info[label] = info
    
    def sim_info_to_json(self, fpath_json):
        # Implement here
        pass

    def run_sims(self): pass  # non-blocking

    def wait_for_completion(self): pass

    # Save/restore the current state of the object
    def save(self, fpath_cache): pass
    def load(self, fpath_cache): pass

# Questions:
# - Where the batch script will be running - locally or on the HPC? (HPC)
# - Where to create netParams - locally or in the job script? (job script)

class SimManagerHPC(SimManager):
    is_running: bool
    
    def _send_file_to_hpc(self, fpath_local, dirpath_hpc) -> str:
        # Returns file path on hpc
        pass
    
    def _ssh_open(self): pass
    def _ssh_close(self): pass

    def _ssh_run_script(self, fpath_script: str, args: Dict): pass
    def _ssh_is_script_running(self, fpath_script) -> bool: pass
    
    def run_sims(self):
        # If run_stims() was already called previously - do nothing
        # It makes run_sims() invariant, so we can break and re-run the process
        if self.is_running == True:
            return
        
        # Save sims_info to json and transfer it to HPC
        fpath_sim_info = ''
        self.sim_info_to_json(fpath_sim_info)
        fpath_sim_info_hpc = self._send_file_to_hpc(fpath_sim_info, dirpath_hpc='')
        
        # Connect to HPC via ssh and run a batch script (in bkg mode)
        fpath_batch_script = ''
        fpath_job_script = ''
        self._ssh_open()
        if self.ssh_is_script_running(fpath_batch_script):
            self._ssh_close()
            raise ValueError('Already running')
        args = {'fpath_sims_info': fpath_sim_info_hpc,
                'fpath_job_script': fpath_job_script}
        self._ssh_run_script(fpath_batch_script, args)
        self._ssh_close()
        
        # We assume that the batch script will run on HPC until completion,
        # so we can break/restart the local code without re-running the batch
        self.is_running = True
    
    