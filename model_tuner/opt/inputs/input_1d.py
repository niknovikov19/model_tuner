from dataclasses import dataclass

import numpy as np

from .input_base import PopInput, NetInput


@dataclass
class PopInput1D(PopInput):
    value: float = 0


@dataclass
class NetInput1D(NetInput):

    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('value')