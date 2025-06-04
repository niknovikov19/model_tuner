import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from model_tuner.opt.regimes import NetRegime1D, NetRegime1DList
from model_tuner.opt.ir_mappers import NetIRMapper1DSlice


def plot_ir_mapping_1d_slice(
        net_ir_mapper: NetIRMapper1DSlice,
        rate_mats: Dict[str, xr.DataArray],
        inp_limits: Dict[str, Tuple[float, float]],
        dirpath_out: Path | str,
        target_rates: Dict[str, float] | None = None,
        r_vis_max: float | None = None,
        inp_vis_max: float | None = None,
        npts: int = 200,
        rbase: float = 0.001,
        rmax: float = 100
        ) -> None:
    
    pop_names = net_ir_mapper.pop_names
    npops = len(pop_names)
    
    # Create output folder
    if isinstance(dirpath_out, str):
        dirpath_out = Path(dirpath_out)
    os.makedirs(dirpath_out, exist_ok=True)

    # Generate a range of output rates for each pop.
    #rates_out_vec = np.geomspace(rbase, rmax, npts) - rbase
    rates_out_vec = np.linspace(rbase, rmax, npts)
    rates_out_mat = np.tile(rates_out_vec, (npops, 1))

    # Convert output rates to NetRegime1DList
    regimes_out = NetRegime1DList.from_regimes_mat(
        pop_names, rates_out_mat
    )

    # Map regimes to inputs
    inputs = [net_ir_mapper.R_to_I(regime) for regime in regimes_out]

    # Extract ou_mean and ou_std matrices from the input list
    ou_mean_mat = np.zeros((npops, npts))
    ou_std_mat = np.zeros((npops, npts))
    for m, inp in enumerate(inputs):
        ou_mean_mat[:, m] = inp.get_pop_inputs_vec('ou_mean')
        ou_std_mat[:, m] = inp.get_pop_inputs_vec('ou_std')
    
    # Map target regime to input
    if target_rates:
        regime_target = NetRegime1D.from_dict(target_rates)
        inp_target = net_ir_mapper.R_to_I(regime_target)

    for n, pop_name in enumerate(pop_names):
        print(f'Plotting I-R mapping for {pop_name}...')

        R = rate_mats[pop_name]
        
        # Slie of the training data
        ou_mean_vec = R.coords['ou_mean'].values
        if R.ndim == 2:
            slicer = net_ir_mapper.pop_IR_mappers[pop_name].slicer
            rr_vec = slicer.get_1d_slice(R, ou_mean_vec)
        else:
            rr_vec = R.values

        # Mask for the training data that was used for fitting
        if inp_limits is not None:
            mask = ((ou_mean_vec >= inp_limits[pop_name][0]) &
                    (ou_mean_vec <= inp_limits[pop_name][1]))
        else:
            mask = np.full_like(ou_mean_vec, True, dtype=bool)

        # I-R mapping result
        ou_mean_vec_hat = ou_mean_mat[n, :]
        ou_std_vec_hat = ou_std_mat[n, :]
        rr_vec_hat = rates_out_vec

        mult = 100
        ou_mean_vec = ou_mean_vec * mult
        ou_mean_vec_hat = ou_mean_vec_hat * mult
        ou_std_vec_hat = ou_std_vec_hat * mult

        plt.figure(111)
        plt.clf()

        plt.plot(ou_mean_vec[~mask], rr_vec[~mask], 'kx')
        plt.plot(ou_mean_vec[mask], rr_vec[mask], 'k.', markersize=6)
        plt.plot(ou_mean_vec_hat, rr_vec_hat, 'r-', linewidth=2)

        if target_rates:
            plt.plot(inp_target[pop_name]['ou_mean'] * mult, 
                     regime_target[pop_name].value, 
                     'b.', markersize=12)

        plt.title(pop_name)
        plt.xlabel(f'OU mean * {mult}')
        plt.ylabel('Rate')

        if inp_vis_max:
            plt.xlim(0, inp_vis_max)
        if r_vis_max:
            plt.ylim(0, r_vis_max)

        #plt.show()
        #plt.draw()

        #print('Saving figure...')
        fpath_fig = dirpath_out / f'{n}_{pop_name}.png'
        plt.savefig(fpath_fig)
