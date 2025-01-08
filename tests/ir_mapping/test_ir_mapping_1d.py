from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model_tuner.sim_manager import SimManagerHPCBatch, SimBatchPaths, SimStatus
from model_tuner.ssh import SSHParams, SSHClient

from model_tuner.opt_base import PopInput, NetInput
from model_tuner.opt_base import PopRegime, NetRegime, NetRegimeList
from model_tuner.opt_base import PopIRMapper, NetIRMapper, NetUCMapper

from model_tuner.opt_base import MapFunc1D, MapFunc1DExp, MapFunc1DSigmoid


def create_map_func_by_name(func_name: str) -> MapFunc1D:
    if func_name == 'exp_1d':
        return MapFunc1DExp()
    if func_name == 'sigmoid_1d':
        return MapFunc1DSigmoid()
    raise ValueError(f'Unknown map function: {func_name}')


@dataclass
class PopInput1D(PopInput):
    value: float

@dataclass
class PopRegime1D(PopRegime):
    value: float


class PopIREmpiricalMapper1D(PopIRMapper):
    def __init__(
            self,
            map_type: Literal['exp_1d', 'sigmoid_1d'] = 'exp_1d'
            ):
        self._map_func = create_map_func_by_name(map_type)
        
    def I_to_R(self, I: PopInput1D) -> PopRegime1D:
        x_out = self._map_func.apply(I.value)
        return PopRegime1D(value=x_out)

    def R_to_I(self, R: PopRegime1D) -> PopInput1D:
        x_in = self._map_func.apply_inv(R.value)
        return PopInput1D(value=x_in)
    
    def fit_from_data(
            self,
            values_in: np.ndarray,
            values_out: np.ndarray
            ) -> None:
        if len(values_in) != len(values_out):
             raise ValueError('Value vectors should have the same length')
        self._map_func.fit(values_in, values_out)


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


class BatchMetricGetterTest(BatchMetricGetter):
    
    def __init__(self):
        self._batch_params = {
            'rxe': np.linspace(0, 10, 20)
        }
        self._pop_names = ['pop1', 'pop2']
        self._gen_surrogate_data()
    
    def _gen_surrogate_data(self):
        self._rate_data = {}
        self._data_gen_par = {
            'pop1': {'a': 1.0, 'b': 0.1, 'k': 10.0, 's': 0.2},
            'pop2': {'a': 2.0, 'b': 0.2, 'k': 20.0, 's': 0.5}
        }
        for pop_name in self._pop_names:
            x = self._batch_params['rxe']
            a, b, k, s = [self._data_gen_par[pop_name][p]
                          for p in ('a', 'b', 'k', 's')]
            y = a * np.exp(b * x) + k + s * np.random.rand(len(x))
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
bmg = BatchMetricGetterTest()

# Batch parameter values that characterize the model input
inp_rates = bmg.get_batch_par_values('rxe')

pop_rates = {}

# Network input-to-regime mapper
net_ir_mapper = NetIRMapper()

for pop_name in bmg.get_pop_names():
    # Request firing rates of a pop (for every batch parameter value)
    pop_rates[pop_name] = bmg.get_pop_rates_batch(pop_name)
    
    # Fit input-to-regime mapping for a pop
    pop_ir_mapper = PopIREmpiricalMapper1D(map_type='exp_1d')
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
    net_input = NetInput(pop_inputs=pop_inputs)
    
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
    
    




