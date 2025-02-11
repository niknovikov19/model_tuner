from typing import List, Tuple, Union

import numpy as np

from .map_func_1d import MapFunc1D


class MapFunc1DSigmoid(MapFunc1D):
    
    @classmethod
    def get_par_names(cls) -> List[str]:
        return ['a', 'b', 'c', 'k']
    
    def f(self, x: np.ndarray, a, b, c, k) -> np.ndarray:
        y = c + a / (1 + np.exp(-k * (x - b)))
        return y
    
    def f_inv(self, y: np.ndarray, a, b, c, k) -> np.ndarray:
        y = y.copy()
        y[(a / (y - c)) < 1] = np.nan
        x = b - np.log(a / (y - c) - 1) / k
        return x
    
    @classmethod
    def _get_fit_bounds(cls) -> Tuple[List[float], List[float]]:
        bounds = {p: (-np.inf, np.inf) for p in ['a', 'b', 'c', 'k']}
        #bounds['c'] = (0, np.inf)
# =============================================================================
#         bounds = {
#             'a': (0.1, 10),
#             'b': (-10, 20),
#             'c': (-10, 20),
#             'k': (0.1, 10)
#         }
# =============================================================================
        low = [bounds[p][0] for p in ['a', 'b', 'c', 'k']]
        high = [bounds[p][1] for p in ['a', 'b', 'c', 'k']]
        return low, high
    
# =============================================================================
#     @staticmethod
#     def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple:
#         k0 = 1 / np.max(np.abs(xx))
#         c0 = np.nanmin(yy)
#         a0 = np.nanmax(yy) - c0
#         th = c0 + a0 / 2
#         mask1 = yy < th
#         mask2 = yy > th
#         n1 = np.argmax(yy[mask1])
#         n2 = np.argmin(yy[mask2])
#         x1, y1 = xx[mask1][n1], yy[mask1][n1]
#         x2, y2 = xx[mask2][n2], yy[mask2][n2]
#         b0 = (x1 + x2) / 2
#         return a0, b0, c0, k0
# =============================================================================
    
    @classmethod
    def _get_first_fit_guess(cls, xx: np.ndarray, yy: np.ndarray) -> Tuple:
        is_increasing = yy[-1] > yy[0]
        
# =============================================================================
#         if is_increasing:
#             c0 = np.nanmin(yy)
#             a0 = np.nanmax(yy) - c0
#         else:
#             c0 = np.nanmax(yy)
#             a0 = np.nanmin(yy) - c0
# =============================================================================

        c0 = np.nanmin(yy)
        a0 = np.nanmax(yy) - c0
    
        # Points closest to the middle threshold 
# =============================================================================
#         th = c0 + a0 / 2
#         mask1 = yy < th
#         mask2 = yy > th
#         n1 = np.argmax(yy[mask1])  # Highest value below threshold
#         n2 = np.argmin(yy[mask2])  # Lowest value above threshold
#         x1, y1 = xx[mask1][n1], yy[mask1][n1]
#         x2, y2 = xx[mask2][n2], yy[mask2][n2]
# =============================================================================
        
        # Interval with the max. slope and its midpoint
        dy = yy[1:] - yy[:-1]
        n = np.nanargmax(dy)
        x01, y01 = xx[n], yy[n]
        x02, y02 = xx[n + 1], yy[n + 1]
        b0 = (x01 + x02) / 2
        
        # Corner points
        x1, y1 = xx[0], yy[0]
        x2, y2 = xx[-1], yy[-1]
        
        for _ in range(5):
            k0 = 4 * (y02 - y01) / (x02 - x01) / a0
            S = lambda x: 1 / (1 + np.exp(-k0 * (x - b0)))
            a0 = (y2 - y1) / (S(x2) - S(x1))
            c0 = y1 - a0 * S(x1)
        
        #k0 = 4 * (y02 - y01) / (x02 - x01) / a0
    
# =============================================================================
#         # Adjust k0 based on monotonicity
#         k0 = 1 / np.max(np.abs(xx))
#         if not is_increasing:
#             k0 = -k0  # Flip sign if decreasing
# =============================================================================
        
        return a0, b0, c0, k0