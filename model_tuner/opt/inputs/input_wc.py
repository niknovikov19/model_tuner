from dataclasses import dataclass
from typing import Dict

import numpy as np

from model_tuner.utils import from_dict_or_dataclass

from .input_1d import PopInput1D, NetInput1D


@dataclass        
class PopInputWC(PopInput1D):
    def __init__(self, I: float = 0, **kwargs):
        super().__init__(value=I, **kwargs)

    @property
    def I(self) -> float:
        return self.value
    
    @I.setter
    def I(self, new_value: float) -> None:
        self.value = new_value


@dataclass
class NetInputWC(NetInput1D):
    """ def __init__(self, pop_inputs: Dict[str, PopInputWC | dict] = None):
        pop_inputs = pop_inputs or {}
        self.pop_inputs = {
            pop_name: from_dict_or_dataclass(pop_input, PopInputWC)
            for pop_name, pop_input in pop_inputs.items()
        } """

    @classmethod
    def _convert_parent(cls, R: NetInput1D) -> 'NetInputWC':
        R.__class__ = NetInputWC
        for r in R.pop_regimes.values():
            r.__class__ = PopInputWC
        return R
