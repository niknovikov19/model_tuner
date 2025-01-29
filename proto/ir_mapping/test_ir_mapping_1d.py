from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.sim_manager import SimManagerHPCBatch, SimBatchPaths, SimStatus
from model_tuner.ssh import SSHParams, SSHClient

from model_tuner.opt.inputs import PopInput1D, NetInput1D
from model_tuner.opt.regimes import PopRegime1D, NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import PopIREmpiricalMapper1D
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
# NetUCMapper

from model_tuner.opt.map_funcs import MapFuncType, MapFunc1D
from model_tuner.opt.map_funcs import MapFunc1DExp, MapFunc1DSigmoid


@dataclass
class RateCalcParams:
    time_range: Tuple[float, float] = (0, None)


class BatchMetricGetter(ABC):
    @abstractmethod
    def get_batch_par_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_batch_par_values(self, par_name: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_pop_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_pop_rates_batch(
            self,
            pop_name: str,
            rate_par: RateCalcParams = RateCalcParams()
            ) -> np.ndarray:
        pass


class BatchMetricGetterSurrogate(BatchMetricGetter):
    
    def __init__(self):
        self._batch_params = {
            'rxe': np.linspace(0, 10, 20)
        }
        self._pop_names = ['pop1', 'pop2']
        self._gen_surrogate_data()
    
    def _gen_surrogate_data(self):
        self._rate_data = {}
        self._data_gen_par = {
            'pop1': {'a': 1.0, 'b': 0.1, 'k': -1.3, 's': 0.2},
            'pop2': {'a': 2.0, 'b': 0.2, 'k': -3.1, 's': 0.5}
        }
        self._positive = True
        for pop_name in self._pop_names:
            x = self._batch_params['rxe']
            a, b, k, s = [self._data_gen_par[pop_name][p]
                          for p in ('a', 'b', 'k', 's')]
            y = a * np.exp(b * x) + k + s * np.random.rand(len(x))
            if self._positive:
                y = np.maximum(y, 0)
            self._rate_data[pop_name] = y
    
    def get_batch_par_names(self) -> List[str]:
        return list(self._batch_params.keys())
    
    def get_batch_par_values(self, par_name: str) -> np.ndarray:
        return self._batch_params[par_name]
    
    def get_pop_names(self) -> List[str]:
        return self._pop_names
    
    def get_pop_rates_batch(
            self,
            pop_name: str,
            rate_par: RateCalcParams = RateCalcParams()
            ) -> np.ndarray:
        return self._rate_data[pop_name]


# Object to request metrics of batch sim results
bmg = BatchMetricGetterSurrogate()

# Batch parameter values that characterize the model input
inp_rates = bmg.get_batch_par_values('rxe')

pop_rates = {}

# Network input-to-regime mapper
net_ir_mapper = NetIREmpiricalMapper1D()

for pop_name in bmg.get_pop_names():
    # Request firing rates of a pop (for every batch parameter value)
    pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
    
    # Fit input-to-regime mapping for a pop
    pop_ir_mapper = PopIREmpiricalMapper1D(
        map_type='exp_1d',
        map_params = {
            'x_positive': False,
            'y_positive': True
        }
    )
    pop_ir_mapper.fit_from_data(inp_rates, pop_rates[pop_name])
    
    net_ir_mapper.set_pop_mapper(pop_name, pop_ir_mapper)


## Apply the resulting network I-R mapper to a range of input rates

pop_names = bmg.get_pop_names()

n_points = 20
rr_inp = np.linspace(0, 10, n_points)
rr_pop = {pop_name: np.zeros(n_points) for pop_name in pop_names}

for n, r_inp in enumerate(rr_inp):    
    # Combine pop inputs to a network input
    pop_inputs = {pop_name: PopInput1D(value=r_inp)
                  for pop_name in pop_names}
    net_input = NetInput1D(pop_inputs=pop_inputs)
    
    # Apply I-R mapping
    net_regime = net_ir_mapper.I_to_R(net_input)
    
    # Store the resulting pop rates
    for pop_name in pop_names:
        rr_pop[pop_name][n] = net_regime.pop_regimes[pop_name].value


## Visualize the results

plt.figure()

for n, pop_name in enumerate(pop_names):
    plt.subplot(1, len(pop_names), n + 1)
    
    # Fitted data produced by the I-R mapper
    plt.plot(rr_inp, rr_pop[pop_name])
    
    # Data used to "learn" the I-R mapping
    plt.plot(inp_rates, pop_rates[pop_name], 'k.')
    
    plt.title(pop_name)
    plt.xlabel('Input rate')
    plt.xlabel('Pop. rate')
