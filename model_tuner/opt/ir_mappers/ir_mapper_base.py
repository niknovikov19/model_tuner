from abc import ABC, abstractmethod
from typing import Dict

from ..inputs import PopInput, NetInput
from ..regimes import PopRegime, NetRegime


class PopIRMapper(ABC):
    @abstractmethod
    def I_to_R(self, I: PopInput) -> PopRegime: pass

    @abstractmethod
    def R_to_I(self, R: PopRegime) -> PopInput: pass

    
class NetIRMapper:
    def __init__(self):
        self.pop_IR_mappers: Dict[str, PopIRMapper] = {}
        
    def set_pop_mapper(self, pop_name: str, mapper: PopIRMapper):
        self.pop_IR_mappers[pop_name] = mapper
    
    @property
    def pop_names(self):
        return list(self.pop_IR_mappers.keys())
        
    def I_to_R(self, I: NetInput) -> NetRegime:
        R = NetRegime()
        for name, I_ in I.pop_inputs.items():
            R.pop_regimes[name] = self.pop_IR_mappers[name].I_to_R(I_)
        return R
    
    def R_to_I(self, R: NetRegime) -> NetInput:
        I = NetInput()
        for name, R_ in R.pop_regimes.items():
            I.pop_inputs[name] = self.pop_IR_mappers[name].R_to_I(R_)
        return I
    
    def __getitem__(self, pop_name: str) -> PopIRMapper:
        return self.pop_IR_mappers[pop_name]

    #def __setitem__(self, pop_name: str, mapper: PopIRMapper):
    #    self.pop_IR_mappers[pop_name] = mapper
