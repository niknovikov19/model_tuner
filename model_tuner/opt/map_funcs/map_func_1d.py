from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit


class MapFunc1D(ABC):
    def __init__(self):
        self.par = {name: np.nan for name in self.get_par_names()}
    
    @staticmethod
    @abstractmethod
    def get_par_names() -> List[str]: pass

    def get_par_vals(self) -> List:
        return [self.par[name] for name in self.get_par_names()]
        
    @staticmethod
    @abstractmethod
    def f(x: Union[float, np.ndarray],
          *args, **kwargs
          ) -> Union[float, np.ndarray]:
        pass
    
    @staticmethod
    @abstractmethod
    def f_inv(x: Union[float, np.ndarray],
              *args, **kwargs
              ) -> Union[float, np.ndarray]:
        pass
    
    def apply(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f(x, **self.par)
    
    def apply_inv(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f_inv(y, **self.par)
    
    @staticmethod
    @abstractmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]: pass
        
    @staticmethod
    @abstractmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple: pass
    
    def fit(self, xx: np.ndarray, yy: np.ndarray, from_prev=False):
        if from_prev:
            par0 = self.get_par_vals()
        else:
            par0 = self._get_first_fit_guess(xx, yy)
        bounds = self._get_fit_bounds()
        try:
            par, _ = curve_fit(self.f, xx, yy, p0=par0,
                               bounds=bounds,
                               nan_policy='omit')
            self.par = {name: par[n]
                        for n, name in enumerate(self.get_par_names())}
        except Exception as e:
            print(f'Fitting failed ({e})')
            self.par = {name: np.nan for name in self.get_par_names()}
