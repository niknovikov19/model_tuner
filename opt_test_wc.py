import matplotlib.pyplot as plt
import numpy as np

from defs_wc import NetInputWC, NetRegimeWC, NetRegimeListWC
from mappers_wc import NetIRMapperWC, NetUCMapperWC
from model_wc import PopParamsWC, ModelDescWC, wc_gain, run_wc_model


def create_test_model_1pop():
    model = ModelDescWC.create_unconn(num_pops=1)
    model.conn[0, 0] = -0.1
    return model

def create_test_model_2pop():
    model = ModelDescWC.create_unconn(num_pops=2)
    model.conn = np.array([[0.1, -0.2], [0.2, -0.1]])
    return model


# Network of Wilson-Cowan populations
#model = create_test_model_1pop()
model = create_test_model_2pop()

pop_names = model.get_pop_names()
npops = len(pop_names)

# Original target regime (vector of pop. firing rates)
rr_base = np.arange(npops) + 1

# P_FR
pfr_vec = np.linspace(0.1, 1.5, 20)

# Target regimes (base * pfr for each pfr)
R0_lst = NetRegimeListWC(
    [NetRegimeWC(pop_names, rr_base * pfr) for pfr in pfr_vec]
)

# I-R mapper, explicitly uses Wilson-Cowan gain functions of populations
ir_mapper = NetIRMapperWC(model)

# Unconnected-to-connected regime mapper
#uc_mapper = NetUCMapperWC(pop_names, 'exp')
uc_mapper = NetUCMapperWC(pop_names, 'sigmoid')
uc_mapper.set_to_identity()

# Params of WC model simulations
sim_par = {'niter': 20, 'dr_mult': 1}

need_plot_iter = 0
need_plot_res = 1

n_iter = 20

for iter_num in range(n_iter):
    print(f'Iter: {iter_num}')
    Rc_lst = R0_lst.copy()
    Rc_prev_lst = Rc_lst.copy()
    Ru_lst = uc_mapper.Rc_to_Ru(Rc_lst)
    for n, Ru in enumerate(Ru_lst):
        Iu = ir_mapper.R_to_I(Ru)
        Rc_new, sim_info = run_wc_model(
            model, Iu, sim_par['niter'], Ru, sim_par['dr_mult']
        )
        Rc_lst[n] = Rc_new
        sim_err = np.abs(sim_info['dr_mat'][:, -1]).max()
        err = np.abs(R0_lst[n].get_pop_rates_vec() - Rc_new.get_pop_rates_vec()).max()
        n_nan = np.sum(np.isnan(Rc_new.get_pop_rates_vec()))
        print(f'Point: {n}, sim_err = {sim_err:.02f}, nan = {n_nan}, '
              f'err = {err:.04f}')
    uc_mapper.fit_from_data(Ru_lst, Rc_lst)
    
    if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
        plt.figure(113)
        plt.clf()
        
        ru_mat = Ru_lst.get_pop_attr_mat('r')
        rc_mat = Rc_lst.get_pop_attr_mat('r')
        rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('r')
        iu_mat = np.full_like(ru_mat, np.nan)
        
        for m in range(ru_mat.shape[1]):
            Ru_ = NetRegimeWC(pop_names, ru_mat[:, m])
            iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_attr_vec('I')
            
        for n, pop in enumerate(pop_names):
            rr_u = ru_mat[n, :]
            rr_c = rc_mat[n, :]
            rr_c_prev = rc_prev_mat[n, :]
            ii_u = iu_mat[n, :]

            plt.subplot(2, npops, n + 1)
            plt.plot(ii_u, rr_u, '.')
            ii_u_ = np.linspace(np.nanmin(ii_u), np.nanmax(ii_u), 200)
            plt.plot(ii_u_, wc_gain(ii_u_, model.pops[pop]))
            plt.xlabel('Iu')
            plt.ylabel('Ru')
            rvis_max = rr_base[n] * pfr_vec.max() * 1.2
            plt.xlim(-3.5, 0)
            plt.ylim(0, rvis_max)
            plt.title(f'pop = {pop}')
            
            plt.subplot(2, npops, npops + n + 1)
            plt.plot(rr_u, rr_c, '.')
            rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
            plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
            plt.plot(rr_u, rr_c_prev, 'kx')
            plt.xlabel('Ru')
            plt.ylabel('Rc')
            plt.xlim(0, rvis_max)
            plt.ylim(0, rvis_max)
        
        plt.draw()
        if need_plot_iter:
            if not plt.waitforbuttonpress():
                break