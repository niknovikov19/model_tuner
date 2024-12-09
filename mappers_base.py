from typing import Dict, List, Tuple, Union, Literal

from defs_base import PopInput, NetInput
from defs_base import PopRegime, NetRegime, NetRegimeList


class PopIRMapper:
    def I_to_R(self, I: PopInput) -> PopRegime: pass
    def R_to_I(self, R: PopRegime) -> PopInput: pass
    
class NetIRMapper:
    def __init__(self):
        self.pop_IR_mappers: Dict[str, PopIRMapper] = {}
        
    def set_pop_mapper(self, pop_name: str, mapper: PopIRMapper):
        self.pop_IR_mappers[pop_name] = mapper
        
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


class NetUCMapper:
    def _Ru_to_Rc(self, Ru: NetRegime) -> NetRegime:
        pass
    
    def _Rc_to_Ru(self, Rc: NetRegime) -> NetRegime:
        pass
    
    def Ru_to_Rc(
            self,
            Ru: Union[NetRegime, NetRegimeList]
            ) -> Union[NetRegime, NetRegimeList]:
        """Unconnected -> connected. """
        if isinstance(Ru, NetRegime):
            return self._Ru_to_Rc(Ru)
        else:
            return NetRegimeList([self._Ru_to_Rc(Ru_) for Ru_ in Ru])
        
    def Rc_to_Ru(
            self,
            Rc: Union[NetRegime, NetRegimeList]
            ) -> Union[NetRegime, NetRegimeList]:
        """Connected -> unconnected. """
        if isinstance(Rc, NetRegime):
            return self._Rc_to_Ru(Rc)
        else:
            return NetRegimeList([self._Rc_to_Ru(Rc_) for Rc_ in Rc])
        
    def fit_from_data(self, Ru: NetRegimeList, Rc: NetRegimeList):
        pass
