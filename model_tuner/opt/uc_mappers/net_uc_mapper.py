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
    
    @abstractmethod        
    def fit_from_data(self, Ru: NetRegimeList, Rc: NetRegimeList):
        pass
