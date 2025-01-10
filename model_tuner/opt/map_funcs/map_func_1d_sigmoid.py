from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


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