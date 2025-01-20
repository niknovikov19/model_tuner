from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from io import IOBase
from pathlib import Path
import pickle

from fs import FS

from data_keeper import DataKeeper
from model_tuner.opt.regimes import NetRegimeWC
from proc_params import SimResultParseParams
#from sim_result import SimResultFile
from sim_result_manager import SimResultManager


def get_sim_regime_wc_direct(
        sim_label: str,
        dirpath_res: str | Path,
        fs: FS
        ) -> NetRegimeWC:
    manager = SimResultManager(dirpath_res, fs)  # maps sim labels to result locations
    res = manager.locate_result(sim_label)  # get info about the result location
    with res.open('rb') as fid:
        sim_result = pickle.load(fid)
    return sim_result['R']  # stored as NetRegimeWC already by the job script


def get_sim_regime_wc_dk(
        sim_label: str,
        dirpath_res_hpc: str | Path,
        dk: DataKeeper,
        fs: FS,
        need_recalc: bool = False,
        ) -> NetRegimeWC:
    
    # Description of the step: extraction of the regime from a simulation result
    step_name = 'parse_sim_result'
    step_par = SimResultParseParams(
        sim_type='WC',
        sim_label=sim_label,
        dirpath_res=dirpath_res_hpc,
        data_type='net_regime'
    )
    proc_chain = {step_name: step_par}
    data_name = 'net_regime'
    
    # If the required data is not in the storage yet or recalculation is required
    if need_recalc or not dk.exists(data_name, proc_chain):
        
        # Read simulation result
        manager = SimResultManager(dirpath_res_hpc, fs)  # maps labels to locations
        res = manager.locate_result(sim_label)  # get info about the result location
        with res.open('rb') as fid:
            sim_result = pickle.load(fid)
    
        # Extract the regime from the sim result and store it locally
        R = sim_result['R']
        dk.store_data(R, data_name, proc_chain)
    
    # Load a previously stored regime
    R = dk.get_data(data_name, proc_chain)
    return R
