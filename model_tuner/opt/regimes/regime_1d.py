from dataclasses import dataclass

import numpy as np

from .regime_base import PopRegime, NetRegime, NetRegimeList


@dataclass        
class PopRegime1D(PopRegime):
    value: float = 0


@dataclass
class NetRegime1D(NetRegime):    

    def get_pop_regime_val(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].value
    
    def get_pop_regimes_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('value')
    
# =============================================================================
#     @classmethod
#     def from_rates(cls, pop_names: List[str], pop_rates: List[float]) -> 'NetRegimeWC':
#         R = NetRegimeWC()
#         for pop_name, r in zip(pop_names, pop_rates):
#             R.pop_regimes[pop_name] = PopRegimeWC(r=r)
#         return R
#     
#     @classmethod
#     def from_rates_dict(cls, pop_rates: Dict[str, float]) -> 'NetRegimeWC':
#         return cls.from_rates(
#             pop_names=list(pop_rates.keys()),
#             pop_rates=list(pop_rates.values())
#         )
# =============================================================================


@dataclass
class NetRegime1DList(NetRegimeList):
    
    def get_pop_regimes_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('value')
