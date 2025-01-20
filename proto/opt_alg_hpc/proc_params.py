from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

#import numpy as np


@dataclass(frozen=True)
class ProcStepParams:
    pass

@dataclass(frozen=True)
class ProcStepParamsEmpty(ProcStepParams):
    pass

@dataclass(frozen=True)
class SourceDataParams(ProcStepParams):
    filepath: Path

# =============================================================================
#     sim_type: str = ''  # what type of simulation produced the result to be parsed
#     sim_label: str
#     dirpath_res: str = ''
#     data_type: str | List[str] = '' # what type(s) of data we obtained by the parsing
# =============================================================================

@dataclass(frozen=True)
class NetSpikesParams(ProcStepParams):
    pop_names: List[str] = None
    combine_cells: bool = True
    time_limits: list = (0, None)
    subtract_t0: bool = True  # make spike times relative to time_limits[0]
    ms: bool = False  # spike times in miliseconds, otherwise - in seconds
    ndigits: int = 6  # spike times are rounded up to this number of digits

@dataclass(frozen=True)
class NetRatesParams(ProcStepParams):
    pop_names: List[str] = None  # None means all populations
    time_limits: Tuple = (0, None)  # None means end of a simulation


from collections.abc import Mapping

@dataclass(frozen=True)
class ProcParamsChain(Mapping):
    
    _chain: Dict[str, ProcStepParams] = field(default_factory=dict)

    def __getitem__(self, key: str) -> ProcStepParams:
        return self._chain[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._chain)

    def __len__(self):
        return len(self._chain)
    
    def __hash__(self):
        return hash(tuple(self._chain.items()))
    
    def add_step(self, name: str, params: ProcStepParams) -> 'ProcParamsChain':
        if name in self._chain:
            raise ValueError(f'Step {name} already exist in the chain')
        chain_new = {**self._chain, name: params}
        return ProcParamsChain(chain_new)
