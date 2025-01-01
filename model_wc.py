from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import numpy as np

from defs_wc import NetInputWC, NetRegimeWC
from defs_base import ModelDesc
from utils import from_dict_or_dataclass


@dataclass
class PopParamsWC:
    mult: float = 10
    gain: float = 1
    thresh: float = 1


@dataclass
class ModelDescWC(ModelDesc):
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
    

def wc_gain(x: float, pop: PopParamsWC) -> float:
    return pop.mult / (1 + np.exp(-pop.gain * (x - pop.thresh)))

def wc_gain_inv(r: float, pop: PopParamsWC) -> float:
    return -np.log(pop.mult / r - 1) / pop.gain + pop.thresh


def _column(x: np.ndarray) -> np.ndarray:
    x = x.squeeze()
    if x.ndim > 1:
        raise ValueError('Input has more than one non-singleton dimension')
    return x.reshape(-1, 1)

def run_wc_model(
        model: ModelDescWC,
        Iext: NetInputWC,
        niter: int,
        R0: NetRegimeWC = None,
        dr_mult: float = 1
        ) -> Tuple[NetRegimeWC, Dict]:
    npops = len(model.pops)
    # External input
    #ii_ext = Iext.get_pop_inputs_vec().reshape(-1, 1)
    ii_ext = Iext.get_pop_attr_vec('I')
    # Initial rates
    if R0 is None:
        rr = np.zeros(npops)
    else:
        rr = R0.get_pop_rates_vec()
    # Simulation details
    info = {'r_mat': np.zeros((npops, niter)),
            'dr_mat': np.zeros((npops, niter))}
    # Main loop
    rr_new = np.zeros(npops)
    for n in range(niter):
        ii = (model.conn @ _column(rr) + ii_ext).flatten()
        for m, pop in enumerate(model.pops.values()):
            rr_new[m] = wc_gain(ii[m], pop)
        drr = rr_new - rr
        rr += dr_mult * drr
        info['r_mat'][:, n], info['dr_mat'][:, n] = rr, drr
    # Result
    R = NetRegimeWC.from_rates(model.get_pop_names(), rr)
    return R, info


# Test of ModelDescWC constructors
if __name__ == '__main__':
    
    pops1 = {
        'pop1': PopParamsWC(mult=11, gain=1, thresh=0.1),
        'pop2': PopParamsWC(mult=22, gain=2, thresh=0.2)
    }
    model1 = ModelDescWC(pops=pops1)
    print(model1)
    
    pops2 = {
        'pop1': {'mult': 11, 'gain': 1, 'thresh': 0.1},
        'pop2': {'mult': 22, 'gain': 2, 'thresh': 0.2},
    }
    model2 = ModelDescWC(pops=pops2)
    print(model2)
    