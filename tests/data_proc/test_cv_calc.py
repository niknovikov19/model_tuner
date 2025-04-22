from dataclasses import dataclass
from pprint import pprint

import numpy as np

from model_tuner.data_proc import data_proc_utils as proc_utils


def gen_cell_spikes(r: float, CV: float, T: float) -> np.ndarray:
    mean_ISI = 1 / r
    spike_times = []
    current_time = 0.0
    while current_time < T:
        if CV == 0:
            ISI = mean_ISI
        else:
            shape = 1 / CV**2
            scale = mean_ISI / shape
            ISI = np.random.gamma(shape, scale)
        current_time += ISI
        if current_time < T:
            spike_times.append(current_time)
    return np.array(spike_times)

@dataclass
class PopState:
    r: float | list[float] | np.ndarray
    CV: float

def gen_pop_spikes(state: PopState, T: float) -> list[np.ndarray]:
    r = state.r
    r = [r] if isinstance(r, float) else r
    pop_spikes = []
    for r_ in r:
        pop_spikes.append(gen_cell_spikes(r_, state.CV, T))
    return pop_spikes

def gen_net_spikes(states: dict[str, PopState], T: float
                   ) -> dict[str, list[np.ndarray]]:
    net_spikes = {}
    for pop_name, state in states.items():
        net_spikes[pop_name] = gen_pop_spikes(state, T)
    return net_spikes


states = {
    'pop1': PopState(r=np.linspace(1, 20, 100),
                     CV=0.7),
    'pop2': PopState(r=np.linspace(1, 50, 200),
                     CV=1.4),
    'pop3': PopState(r=np.linspace(1, 10, 10),
                     CV=1)
}

T = 3

t_range = (0, T)
nspikes_min = 5

net_spikes = gen_net_spikes(states, T=T)

cvs = proc_utils.calc_net_cvs(
    net_spikes, t_range, nspikes_min=nspikes_min
)

for pop_name, cv in cvs.items():
    print(f'{pop_name}: CV = {cv:.02f}')