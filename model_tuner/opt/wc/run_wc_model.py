import numpy as np
from typing import Dict, Tuple

from ..inputs.input_wc import NetInputWC
from ..regimes.regime_wc import NetRegimeWC

from .wc_model_defs import ModelDescWC
from .wc_gain_func import wc_gain


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