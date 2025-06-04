from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr

from model_tuner.opt.regimes import (
    NetRegimeWC, NetRegimeWCList,
    NetRegime1DList
)
from model_tuner.opt.ir_mappers import NetIRMapperWC
from model_tuner.opt.uc_mappers import NetUCMapper1D
from model_tuner.opt.wc import ModelDescWC
from model_tuner.opt.wc import wc_gain, run_wc_model

from model_tuner.main import (
    UCMapFitParams,
    plot_opt_iteration_pop
)
from model_tuner.utils import load_yaml

from model_tuner.opt.uc_optimizer import (
    OptStrategy, OptStrategyParams, UCOptimizer
)

warnings.filterwarnings('ignore')


def load_uc_map_params() -> UCMapFitParams:
    """Load UC map params. """
    dirpath_base = Path(
        r'D:\WORK\Salvador\repo\model_tuner\test_data\main'
        r'\test_opt_A1_hpc_batch_qsub\experiments'
        r'\test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=1'
    )
    uc_map_params: UCMapFitParams = load_yaml(
        dirpath_base / 'uc_map_params.yaml',
        data_class=UCMapFitParams
    )
    return uc_map_params


def plot_rc_data(
        Rc_data: xr.DataArray,   # (pop x pfr x iter)
        rr_base: List[float] | np.ndarray,   # (pop x 1)
        pops_vis: List[str] | None = None,
        show_xlabel: bool = True,
        show_ylabel: bool = True
        ) -> None:

    pop_names = Rc_data.pop.values
    pfr_vec = Rc_data.pfr.values
    niter = Rc_data.iter.size

    if not pops_vis:
        pops_vis = pop_names
    npops_vis = len(pops_vis)
    
    rr_base = [rr_base[list(pop_names).index(pop)] for pop in pops_vis]

    for pop_num, pop in enumerate(pops_vis):
        #plt.subplot(1, npops_vis, pop_num + 1)
        for pfr in pfr_vec:
            rc_vec = Rc_data.sel(pop=pop, pfr=pfr).values
            plt.plot(rc_vec, label=f'pfr={pfr:.02f}')
            plt.plot([0, niter], [pfr * rr_base[pop_num]] * 2, 'k--')
        #plt.plot([0, niter], [rr_base[pop_num]] * 2, 'k--')
        #plt.legend()
        if show_xlabel:
            plt.xlabel('Iteration')
        if show_ylabel:
            plt.ylabel('Rc')
        plt.title(pop)


def plot_uc_result(
        uc_optimizer: UCOptimizer,
        iters_vis: List[int],
        ru_limits: Tuple[float, float] = (None, None),
        nx: int = 2,
        ny: int = 2,
        ) -> None:

    Rc = uc_optimizer.step_data['Rc']
    Ru = uc_optimizer.step_data['Ru']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(12, 8))

    for pop in range(len(uc_optimizer.pop_names)):
        plt.subplot(nx, ny, pop + 1)

        for color_num, iter in enumerate(iters_vis):
            col = colors[color_num % len(colors)]
            plot_opt_iteration_pop(
                pop_names=uc_optimizer.pop_names,
                pop_name_vis=uc_optimizer.pop_names[pop],
                uc_mapper=uc_optimizer.uc_mappers[iter],
                Ru_lst=NetRegime1DList.from_xr(Ru.sel(iter=iter)), 
                Rc_lst=NetRegime1DList.from_xr(Rc.sel(iter=iter)), 
                ru_limits=ru_limits, color=col
            )
            #cc = {'pop': pop, 'iter': iter}
            #plt.plot(Ru.isel(**cc), Rc.isel(**cc), 'x', markersize=8, color=col)
        
        plt.plot([10, 10], [0, 10], 'k--')
        Rc0 = uc_optimizer.Rc0
        for n in range(Rc0.sizes['pfr']):
            plt.plot([0, 15], [Rc0.isel(pop=pop, pfr=n)] * 2, 'k--')

        if pop >= nx * (ny - 1):
            plt.xlabel('Ru')
        else:
            plt.xlabel('')
            plt.xticks([])

        if (pop % nx) == 0:
            plt.ylabel('Rc')
        else:
            plt.ylabel('')

        plt.title(f'pop: {pop}')
    
    plt.show()


def run_opt_experiment(
        model: ModelDescWC,
        uc_optimizer: UCOptimizer,
        sim_par: Dict,
        uc_mapper_0: NetUCMapper1D | None = None,
        verbose=False
        ) -> None:
    
    uc_optimizer.begin(uc_mapper_0)
    
    for iter_num in tqdm(range(1, uc_optimizer.n_iter)):
    #for iter_num in range(1, uc_optimizer.n_iter):
        if verbose:
            print(f'Iter: {iter_num}')

        # Rc0 -> Ru
        Ru_lst = uc_optimizer.suggest_Ru()
        NetRegimeWCList._convert_parent(Ru_lst)

        # Run the model for each pfr value
        Rc_lst = NetRegimeWCList()
        for n, Ru in enumerate(Ru_lst):
            # Ru -> Iu
            Iu = uc_optimizer.ir_mapper.R_to_I(Ru)
            # Run model with Iu and get the resulting Rc
            Rc, sim_info = run_wc_model(
                model, Iu, sim_par['niter'], Ru, sim_par['dr_mult']
            )
            Rc_lst.append(Rc)
        
        # Store Ru and Rc
        uc_optimizer.store_sim_result(Ru_lst, Rc_lst)
        
        # Fit UC mapping
        uc_optimizer.fit_uc_mapper()

        # Proceed to the next iteration
        uc_optimizer.next_iter()


# WC model
model = ModelDescWC.create_unconn(num_pops=2)

# Connectivity and target rate vector
# 1. Converges to a wrong point
model.conn = np.array([[0.1, -0.2], [0.2, 0]])
rr_base = [3, 6]
# 2. Blows up
#model.conn = np.array([[0.7, -0.2], [0.2, 0]])
#rr_base = [3, 6]

# Target regime multipliers
pfr_vec = np.array([0.4, 0.6, 0.8, 1, 1.2])

# Parameters of WC "simulation"
sim_par = {'niter': 20, 'dr_mult': 1}

# UC map params
uc_map_params = load_uc_map_params()
uc_map_params.pop_names = model.get_pop_names()
#uc_map_params.map_type = 'sigmoid_1d_line'
uc_map_params.map_type = 'richards_1d'
uc_map_params.fit_param_bounds = {
    #'m': (0.2, np.inf)
}

# Number of iterations
n_iter = 15

#iters_vis = np.arange(n_iter, step=4)
iters_vis = [14]

# Optimization strategy
alpha = 0.5
opt_strategy_par = OptStrategyParams(
    opt_strategy = OptStrategy.STEP_TO_NEW,
    alpha_Rc=alpha,
    alpha_Ru=alpha,
    alpha_mult_Rc=0.8,
    alpha_mult_Ru=0.8,
    alpha_min=0.01,
    #dmax_rot=0.5,
    steps_by_pop=1,
    auto_decrease_step=1
)

need_plot_conv = 0
need_plot_uc = 1


# I-R mapper, explicitly uses WC gain functions
ir_mapper = NetIRMapperWC(model)

uc_optimizers: List[UCOptimizer] = []

#wmult_vec = [1, 2, 3, 4]
wmult_vec = [2, 2.25, 2.5]

logging.basicConfig(level=logging.ERROR, force=True)
#logging.basicConfig(level=logging.INFO, force=True)

#plt.ion()

model_cur = deepcopy(model)

# Initialize UC optimizer
uc_optimizer = UCOptimizer(
    uc_map_params, model_cur.get_pop_names(), rr_base, pfr_vec,
    n_iter, ir_mapper, opt_strategy_par
)

# Main simulation-optimization loop
try:
    run_opt_experiment(model_cur, uc_optimizer, sim_par, verbose=0)
except Exception as e:
    print(e)

# Plot Rc convergence
if need_plot_conv:
    plt.figure(figsize=(12, 8))
    for n in range(model.npops):
        plt.subplot(1, 2, n + 1)
        plot_rc_data(uc_optimizer.step_data['Rc'], rr_base,
                    pops_vis=[f'pop{n}'],
                    show_xlabel=(n in (2, 3)),
                    show_ylabel=(n in (0, 2))
                    )
    plt.show()

# Plot UC mapping
if need_plot_uc:
    plot_uc_result(
        uc_optimizer, iters_vis, ru_limits=(0, 15), nx=1
    )

#input('Press any key to continue...')