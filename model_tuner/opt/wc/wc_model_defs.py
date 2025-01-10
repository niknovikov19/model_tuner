from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from model_tuner.utils import from_dict_or_dataclass


@dataclass
class PopParamsWC:
    mult: float = 10
    gain: float = 1
    thresh: float = 1


@dataclass
class ModelDescWC:
    """Description of a Wilson-Cowan model. """
    
    pops: Dict[str, PopParamsWC] = field(default_factory=dict)
    conn: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __init__(
            self,
            pops: Dict[str, PopParamsWC | dict] = None,
            conn: np.ndarray = None
            ):
        pops = pops or {}
        self.pops = {
            pop_name: from_dict_or_dataclass(par, PopParamsWC)
            for pop_name, par in pops.items()
        }
        self.conn = conn or np.array([])
    
    def get_pop_names(self) -> List[str]:
        return list(self.pops.keys())
    
    @classmethod
    def create_unconn(cls, num_pops: int) -> 'ModelDescWC':
        model = ModelDescWC()
        for n in range(num_pops):
            model.pops[f'pop{n}'] = PopParamsWC()
        model.conn = np.zeros((num_pops, num_pops))
        return model
    
    @classmethod
    def create_random(cls, num_pops: int) -> 'ModelDescWC':
        model = ModelDescWC.create_unconn(num_pops)
        model.conn = 2 * np.random.rand(num_pops, num_pops) - 1
        return model    
