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
        Ru_prev_lst: NetRegime1DList | None = None,
        Rc_prev_lst: NetRegime1DList | None = None,
        Rc0_lst: NetRegime1DList | None = None,
        ru_limits: tuple[float | None, float | None] = (None, None),
        **kwargs
        ) -> None:
    
    n = pop_names.index(pop_name_vis)
    
    # Ru and Rc from the current step
    ru_mat = Ru_lst.get_pop_attr_mat('value')
    rc_mat = Rc_lst.get_pop_attr_mat('value')
    rr_u = ru_mat[n, :]
    rr_c = rc_mat[n, :]
    
    # Ru from the previous step
    if Ru_prev_lst is None:
        Ru_prev_lst = Ru_lst
    ru_prev_mat = Ru_prev_lst.get_pop_attr_mat('value')
    rr_u_prev = ru_prev_mat[n, :]
    # Rc from the previous step
    if Rc_prev_lst:
        rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('value')
        rr_c_prev = rc_prev_mat[n, :]
    
    # Target Rc
    Rc0_lst = Rc0_lst or Rc_lst
    rc0_mat = Rc0_lst.get_pop_attr_mat('value')
    rr_c0 = rc0_mat[n, :]
    rr_u0 = np.full_like(rr_c0, np.nan)

    # Inverse mapping: Rc0 -> Ru
    if uc_mapper:
        for m, rc0 in enumerate(rr_c0):
            try:
                rr_u0[m] = uc_mapper[pop_name_vis].Rc_to_Ru(rc0).value
            except Exception as e:
                print(f'Inverse mapping failed for {pop_name_vis} (point {m}): {e}')

    # Find min and max Ru
    rr_u_all = np.hstack([rr_u, rr_u_prev, rr_u0]).ravel()
    ru_limits = list(ru_limits)
    if ru_limits[0] is None:
        ru_limits[0] = np.nanmin(rr_u_all)
    if ru_limits[1] is None:
        ru_limits[1] = np.nanmax(rr_u_all)
    
    # Plot the inverse mapping
    if uc_mapper:
        for ru0, rc0 in zip(rr_u0, rr_c0):
            if not np.isnan(ru0):
                plt.plot(ru_limits, [rc0, rc0], 'k--')
                plt.plot(ru0, rc0, 'ko')

    # Plot (Ru, Rc) from the current step
    plt.plot(rr_u, rr_c, '.', markersize=8, **kwargs)

    # Plot the forward mapping
    if uc_mapper:
        rr_u_ = np.linspace(ru_limits[0], ru_limits[1], 200)
        rr_c_ = [Rc.value for Rc in uc_mapper[pop_name_vis].Ru_to_Rc(rr_u_)]
        plt.plot(rr_u_, rr_c_, **kwargs)

    # Plot (Ru, Rc) from the previous step
    if Rc_prev_lst:
        plt.plot(rr_u_prev, rr_c_prev, 'kx')

    plt.xlabel('Ru')
    plt.ylabel('Rc')
    #plt.xlim(0, rr_c0.max() * 2)
    #plt.ylim(0, rr_c0.max() * 1.2)
    plt.title(pop_name_vis)
    
    plt.draw()        