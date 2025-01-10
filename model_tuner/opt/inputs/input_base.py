from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass      
class PopInput:
    pass


@dataclass
class NetInput:
    pop_inputs: Dict[str, PopInput] = field(default_factory=dict)
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(I, attr) for I in self.pop_inputs.values()])
