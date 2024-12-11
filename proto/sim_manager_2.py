from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Literal, Any

from netpyne.specs import NetParams


######################################################
# This section doesn't know anything about HPC stuff.
# It is mine, and I will extend this code.
# But it is not a part of any other project yet, so we can change notations
# if we decide that smth else is needed at this level of abstraction.

class SimStatus(Enum):
    """Exists on my level of code, doesn't know about HPC stuff."""
    DONE: auto()
    ERROR: auto()
    WAITING: auto()

class SimManager:
    # An object of this class handles multiple simulations (run, get result, ...)
    # This is an abstract base class, doesn't know about HPC stuff.
    # In terms of HPC, simulation == job. A different term is used to avoid
    # confusion between different levels of code.
    # SimManager objects are not intended to stay on-line. There should be possible to 
    # run a simulation, and later on - to get its status/result by its label
    # from a different instance of SimManager (e.g. after re-running the main script).
    
    def __init__(self):
        self.sims = {}  # label: params
    
    # To be implemented in child classes
    def get_sim_status(self, label: str) -> SimStatus: pass
    def _run_sim(self, label: str, params: Dict) -> None: pass
    
    def run_simulation(self, label: str, params: Dict) -> SimStatus:
        """The method is invariant: calling it twice is equivalent to calling it once."""
        if label not in self.sims:
            self.sims[label] = params
            # TODO: take a subset of params to be passed via _rum_sim()
            # Large stuff will be passed via a shared json file
            self._run_sim(label, params)  # Actual running
        # TODO: check params consistency (params == self.sims[label])
        return self.get_sim_status(label)


######################################################
# This section implements HPC-specific stuff.

class HPCJobStatus(Enum):
    PENDING = auto()  # submitted to HPC queue, not running yet
    RUNNING = auto()  # job script has started
    FINISHED_OK = auto()  # job script finished with a good result
    FINISHED_ERR = auto()  # job script finished with a bad result
    CRASHED = auto()  # job script started, not finished, but it is
                      # not in a job list anymore (e.g. by qstat)
                      # (if complicated - can implement it later)
    VOID = auto()  # this job is not in use
    
class HPCJobStatusManager:
    # The idea is to have a class, whose methods could be called both
    # on the server side (my laptop) or from a job script (on a node).
    # The easiest way of implementing such global access is to have
    # a status file accessible from everywhere.
    def get_job_status(self, label: str) -> HPCJobStatus:
        # Implementation here...
        # If the job label is encountered for the first time, return HPCJobStatus.VOID
        pass
    def set_job_status(self, label: str, status: HPCJobStatus) -> None:
        # Implementation here...
        # Maybe check the logic here, states are supposed to come in the order:
        # VOID -> PENDING -> RUNNING -> FINISHED_OK|FINISHED_ERR|CRASHED -> VOID
        # If trying to set an unexpected status - generate an exception
        pass

class SimManagerHPC(SimManager):
    def __init__(self):
        self.jsm = HPCJobStatusManager()
        super().init()
    
    def get_sim_status(self, label: str) -> SimStatus:
        """Summarize HPC-specific stuff into a generic status. """
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        hpc_job_status = self.jsm.get_job_status(label)
        if ((hpc_job_status == HPCJobStatus.PENDING) or
            (hpc_job_status == HPCJobStatus.RUNNING)):
            return SimStatus.WAITING
        elif hpc_job_status == HPCJobStatus.FINISHED_OK:
            return SimStatus.DONE
        elif ((hpc_job_status == HPCJobStatus.FINISHED_ERR) or
              (hpc_job_status == HPCJobStatus.CRASHED)):
            return SimStatus.ERROR
        elif hpc_job_status == HPCJobStatus.VOID:
            raise ValueError(f'Simulation {label} is not associated with an HPC job')
    
    def _run_sim(self, label: str, params: Dict) -> None:
        hpc_job_status = self.jsm.get_job_status(label)
        if hpc_job_status != HPCJobStatus.VOID:
            raise ValueError(f'HPC job {label} already exists, cannot re-run')
        # Set PENDING status before the actual submission to avoid interference
        # with the job script that will try to set the RUNNING status 
        self.jsm.set_job_status(HPCJobStatus.PENDING)
    
        # Do the actual submission here...
        # Job script should receive label and params.


######################################################
# This is the code that will be running on HPC nodes

def job_script(label: str, params: Dict) -> None:
    # label and params are passed frfom SimManagerHPC._run_sim()
    
    jsm = HPCJobStatusManager()
    jsm.set_job_status(label, HPCJobStatus.RUNNING)
    
    # The job body here...
    # Read configs, simulate, save outputs
    result_ok = True
    
    if result_ok:
        jsm.set_job_status(label, HPCJobStatus.FINISHED_OK)
    else:
        jsm.set_job_status(label, HPCJobStatus.FINISHED_ERR)
    
    