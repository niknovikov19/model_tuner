from typing import Dict, List, Tuple, Union, Literal

import numpy as np
from scipy.optimize import curve_fit


class MapFunc1D:
    def __init__(self):
        self.par = {name: np.nan for name in self.get_par_names()}
    
    @staticmethod
    def get_par_names() -> List[str]: pass

    def get_par_vals(self) -> List:
        return [self.par[name] for name in self.get_par_names()]
        
    @staticmethod
    def f(x: Union[float, np.ndarray],
          *args, **kwargs
          ) -> Union[float, np.ndarray]:
        pass
    
    @staticmethod
    def f_inv(x: Union[float, np.ndarray],
              *args, **kwargs
              ) -> Union[float, np.ndarray]:
        pass
    
    def apply(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f(x, **self.par)
    
    def apply_inv(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f_inv(y, **self.par)
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]: pass
        
    @staticmethod
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

class MapFunc1DExp(MapFunc1D):    
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'k']

    @staticmethod
    def f(x: Union[float, np.ndarray],
          a, b, k
          ) -> Union[float, np.ndarray]:
        y = a * np.exp(b * x) + k
        #y[y < 0] = np.nan
        return y
    
    @staticmethod
    def f_inv(y: Union[float, np.ndarray],
              a, b, k
              ) -> Union[float, np.ndarray]:
        x = np.log((y - k) / a) / b
        #x[x < 0] = np.nan
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        return [0, 0, -100], [np.inf, np.inf, np.inf]
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        b0 = 1.0
        k0 = np.nanmin(yy) - 0.1 * np.abs(np.nanmin(yy))
        a0 = (np.nanmax(yy) - k0) / np.exp(b0 * np.nanmax(xx))
        return a0, b0, k0
    
class MapFunc1DSigmoid(MapFunc1D):
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'c', 'k']
    
    @staticmethod
    def f(x: Union[float, np.ndarray],
          a, b, c, k
          ) -> Union[float, np.ndarray]:
        y = c + a / (1 + np.exp(-k * (x - b)))
        return y
    
    @staticmethod
    def f_inv(y: Union[float, np.ndarray],
              a, b, c, k
              ) -> Union[float, np.ndarray]:
        x = b - np.log(a / (y - c) - 1) / k
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        bounds = {
            'a': (0.1, 10),
            'b': (-10, 20),
            'c': (-10, 20),
            'k': (0.1, 10)
        }
        low = [bounds[p][0] for p in ['a', 'b', 'c', 'k']]
        high = [bounds[p][1] for p in ['a', 'b', 'c', 'k']]
        return low, high
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        k0 = 1
        c0 = np.nanmin(yy)
        a0 = np.nanmax(yy) - c0
        th = c0 + a0 / 2
        n1 = np.argmax(yy[yy < th])
        n2 = np.argmin(yy[yy > th])
        x1, y1 = xx[n1], yy[n1]
        x2, y2 = xx[n2], yy[n2]
        b0 = (x1 + x2) / 2
        return a0, b0, c0, k0
