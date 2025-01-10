from dataclasses import dataclass
from typing import Dict

import numpy as np

from model_tuner.utils import from_dict_or_dataclass

from .input_base import PopInput, NetInput


@dataclass
class PopInputWC(PopInput):
    I: float = 0


@dataclass
class NetInputWC(NetInput):
    def __init__(self, pop_inputs: Dict[str, PopInputWC | dict] = None):
        pop_inputs = pop_inputs or {}
        self.pop_inputs = {
            pop_name: from_dict_or_dataclass(pop_input, PopInputWC)
            for pop_name, pop_input in pop_inputs.items()
        }
        
    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('I')
