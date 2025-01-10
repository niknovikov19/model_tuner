import numpy as np

from .wc_model_defs import PopParamsWC


def wc_gain(x: float, pop: PopParamsWC) -> float:
    return pop.mult / (1 + np.exp(-pop.gain * (x - pop.thresh)))

def wc_gain_inv(r: float, pop: PopParamsWC) -> float:
    return -np.log(pop.mult / r - 1) / pop.gain + pop.thresh