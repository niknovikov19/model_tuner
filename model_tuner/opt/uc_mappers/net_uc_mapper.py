from abc import ABC, abstractmethod
from typing import Union

from ..regimes import NetRegime, NetRegimeList


class NetUCMapper(ABC):
    @abstractmethod
    def _Ru_to_Rc(self, Ru: NetRegime) -> NetRegime:
        pass
    
    @abstractmethod
    def _Rc_to_Ru(self, Rc: NetRegime) -> NetRegime:
        pass
    
    def Ru_to_Rc(
            self,
            Ru: NetRegime | NetRegimeList
            ) -> NetRegime | NetRegimeList:
        """Unconnected -> connected. """
        if isinstance(Ru, NetRegime):
            Rc = self._Ru_to_Rc(Ru)
        else:
            Rc = [self._Ru_to_Rc(Ru_) for Ru_ in Ru]
        return type(Ru)(Rc)   # convert Rc to the same type as Ru
        
    def Rc_to_Ru(
            self,
            Rc: NetRegime | NetRegimeList
            ) -> NetRegime | NetRegimeList:
        """Connected -> unconnected. """
        if isinstance(Rc, NetRegime):
            Ru = self._Rc_to_Ru(Rc)
        else:
            Ru = [self._Rc_to_Ru(Rc_) for Rc_ in Rc]
        return type(Rc)(Ru)   # convert Ru to the same type as Rc
    
    @abstractmethod        
    def fit_from_data(self, Ru: NetRegimeList, Rc: NetRegimeList):
        pass
