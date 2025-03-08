import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import NetIREmpiricalMapper1D
from model_tuner.opt.uc_mappers import NetUCMapper1D


def plot_opt_iteration(
        pop_names: Tuple[str, ...],
        ir_mapper: NetIREmpiricalMapper1D,
        uc_mapper: NetUCMapper1D,
        Ru_lst: NetRegime1DList,
        Rc_lst: NetRegime1DList,
        Rc_prev_lst: NetRegime1DList,
        Rc0_lst: NetRegime1DList
        ) -> None:
    
    plt.clf()
    
    ru_mat = Ru_lst.get_pop_attr_mat('value')
    rc_mat = Rc_lst.get_pop_attr_mat('value')
    rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('value')
    rc0_mat = Rc0_lst.get_pop_attr_mat('value')

    npops = len(pop_names)

    iu_mat = np.full_like(ru_mat, np.nan)            
    for m in range(ru_mat.shape[1]):
        Ru_ = NetRegime1D.from_values(pop_names, ru_mat[:, m])
        iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_inputs_vec()
        
    for n, pop in enumerate(pop_names):
        rr_u = ru_mat[n, :]
        rr_c = rc_mat[n, :]
        rr_c_prev = rc_prev_mat[n, :]
        rr_c0 = rc0_mat[n, :]
        ii_u = iu_mat[n, :]

        plt.subplot(2, npops, n + 1)
        plt.plot(ii_u, rr_u, '.')
        plt.xlabel('Iu')
        plt.ylabel('Ru')
        plt.title(f'pop = {pop}')
        
        plt.subplot(2, npops, npops + n + 1)
        plt.plot(rr_u, rr_c, '.')
        rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
        plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
        plt.plot(rr_u, rr_c_prev, 'kx')
        plt.xlabel('Ru')
        plt.ylabel('Rc')
        plt.xlim(0, rr_c0.max() * 2)
        plt.ylim(0, rr_c0.max() * 1.2)
    
    plt.draw()        