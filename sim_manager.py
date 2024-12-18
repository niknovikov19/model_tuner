from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass, fields
from enum import Enum, auto
import json
from typing import Dict, List, Tuple, Union, Literal, Any


class SimStatus(Enum):
    NEED_PUSH: auto()
    WAITING: auto()
    DONE: auto()
    ERROR: auto()

class SimManager(ABC):
    """
    An object of this class handles multiple simulations (run, get result, ...)
    SimManager objects are not intended to stay on-line.
    There should be possible to run a simulation, and later on 
      to get its status or result by its label from a different instance
      of SimManager (e.g. after re-running the main script).
    """
    
    @dataclass
    class SimInfo:
        params: Dict
        status: SimStatus
    
    def __init__(self):
        self.sims: Dict[str, 'SimManager.SimInfo'] = {}
    
    @abstractmethod
    def _update_sim_status(self, label: str) -> None: pass

    @abstractmethod
    def _ready_for_request(self) -> bool: pass
    
    @abstractmethod
    def _init_sim_request(self, label: str, params: Dict, push_now: bool) -> None: pass

    @abstractmethod
    def push_all_requests(self) -> None: pass

    def get_sim_status(self, label: str) -> SimStatus:
        if label not in self.sims:
            raise ValueError(f'Simulation {label} does not exist')
        self._update_sim_status(label)
        return self.sims[label].status
    
    def add_sim_request(
            self,
            label: str,
            params: Dict,
            push_now: bool = False
            ) -> SimStatus:
        """The method is invariant: calling it twice is equivalent to calling it once."""
        if label in self.sims:
            # Check params consistency
            if params != self.sims[label]:
                raise ValueError('Repeated request with different params')
        else:
            # Add new simulation request
            if not self._ready_for_request():
                raise ValueError('Busy, not ready for new requests')
            self.sims[label] = self.SimInfo(
                params=params, status=SimStatus.NEED_PUSH)
            self._init_sim_request(label, params, push_now)        
        return self.get_sim_status(label)