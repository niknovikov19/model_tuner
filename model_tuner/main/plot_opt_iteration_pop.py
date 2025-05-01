import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
from model_tuner.opt.uc_mappers import NetUCMapper1D


def plot_opt_iteration_pop(
        pop_names: Tuple[str, ...],
        pop_name_vis: str,
        uc_mapper: NetUCMapper1D,
        Ru_lst: NetRegime1DList,
        Rc_lst: NetRegime1DList,
        Rc_prev_lst: NetRegime1DList = None,
        Rc0_lst: NetRegime1DList = None,
        ru_limits: Tuple[float, float] = (None, None),
        **kwargs
        ) -> None:
    
    n = pop_names.index(pop_name_vis)
    
    ru_mat = Ru_lst.get_pop_attr_mat('value')
    rc_mat = Rc_lst.get_pop_attr_mat('value')
    rr_u = ru_mat[n, :]
    rr_c = rc_mat[n, :]

    ru_limits = list(ru_limits)
    ru_limits[0] = ru_limits[0] or np.nanmin(rr_u)
    ru_limits[1] = ru_limits[1] or np.nanmax(rr_u)
    
    if Rc_prev_lst:
        rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('value')
        rr_c_prev = rc_prev_mat[n, :]
    
    Rc0_lst = Rc0_lst or Rc_lst
    rc0_mat = Rc0_lst.get_pop_attr_mat('value')
    rr_c0 = rc0_mat[n, :]

    plt.plot(rr_u, rr_c, 'k.', markersize=8)

    if uc_mapper:
        rr_u_ = np.linspace(ru_limits[0], ru_limits[1], 200)
        plt.plot(rr_u_,
                 uc_mapper._map_funcs[pop_name_vis].apply(rr_u_),
                 **kwargs)

    if Rc_prev_lst:
        plt.plot(rr_u, rr_c_prev, 'kx')

    plt.xlabel('Ru')
    plt.ylabel('Rc')
    #plt.xlim(0, rr_c0.max() * 2)
    #plt.ylim(0, rr_c0.max() * 1.2)
    plt.title(pop_name_vis)
    
    plt.draw()        