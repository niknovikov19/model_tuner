from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .input_base import PopInput, NetInput


@dataclass
class PopInputND(PopInput):
    vars: Dict[str, float] = field(default_factory=dict)

    @property
    def var_names(self) -> List[str]:
        return list(self.vars.keys())
    
    def as_array(self) -> np.ndarray:
        return np.array(list(self.vars.values()))        
    
    @classmethod
    def from_array(
            cls,
            arr: np.ndarray,
            var_names: List[str]
            ) -> 'PopInputND':
        return cls(vars={vn: v for vn, v in zip(var_names, arr)})
    
    def __getitem__(self, var_name):
        return self.vars[var_name]


@dataclass
class NetInputND(NetInput):

    def get_pop_inputs_vec(self, var_name: str) -> np.ndarray:
        return np.array([
            pop_inp.vars[var_name] for pop_inp in self.pop_inputs.values()
        ])
    
    def to_values_dict(self) -> Dict[str, Dict[str, float]]:
        vd = {}
        for pop_name, pop_inp in self.pop_inputs.items():
            vd[pop_name] = pop_inp.vars
        return vd
