from dataclasses import dataclass
from typing import Dict

import numpy as np

from .input_base import PopInput, NetInput


@dataclass
class PopInput1D(PopInput):
    value: float = 0

    def is_valid(self) -> bool:
        return not np.isnan(self.value)


@dataclass
class NetInput1D(NetInput):

    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('value')
    
    def to_values_dict(self) -> Dict[str, float]:
        vd = {}
        for pop_name, pop_inp in self.pop_inputs.items():
            vd[pop_name] = pop_inp.value
        return vd