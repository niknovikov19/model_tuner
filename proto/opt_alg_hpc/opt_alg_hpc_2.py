from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from io import IOBase
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from data_keeper import DataKeeper
from filesys import FileSystem, FileSystemLocal
from model_tuner.opt.regimes import NetRegimeWC
from proc_params import ProcStepParams
from sim_res_desc import SimResultDesc, SimResultDescPKL


def locate_sim_res_wc(sim_label: str, dirpath_res: str | Path) -> SimResultDesc:
    fname_res = f'{sim_label}.pkl'
    return SimResultDescPKL(fpath_pkl=Path(dirpath_res) / fname_res)

def locate_sim_res_batchtools(sim_label: str, dirpath_res: str | Path) -> SimResultDesc:
    fname_res = f'result_{sim_label}.pkl'
    return SimResultDescPKL(fpath_pkl=Path(dirpath_res) / fname_res)


def get_sim_regime_wc_direct(
        sim_label: str,
        dirpath_res_hpc: str | Path,
        fs: FileSystem = FileSystemLocal()
        ) -> NetRegimeWC:
    sim_res_desc = locate_sim_res_wc(sim_label, dirpath_res_hpc)
    with fs.open(sim_res_desc.fpath_pkl, 'rb') as fid:
        sim_result = pickle.load(fid)
    return sim_result['R']  # stored as NetRegimeWC already by the job script


@dataclass(frozen=True)
class SimResultParseParams(ProcStepParams):
    sim_type: str = ''  # what type of simulation produced the result to be parsed
    sim_label: str
    dirpath_res: str = ''
    data_type: str | List[str] = '' # what type(s) of data we obtained by the parsing
    

def get_sim_regime_wc_dk(
        sim_label: str,
        dirpath_res_hpc: str | Path,
        dk: DataKeeper,
        fs: FileSystem = FileSystemLocal(),
        need_recalc: bool = False,
        ) -> NetRegimeWC:
    
    # Description of the step: extraction of the regime for a sim result
    step_name = 'parse_sim_result'
    step_par = SimResultParseParams(
        sim_type='WC',
        sim_label=sim_label,
        dirpath_res = dirpath_res_hpc,
        data_type = 'net_regime'
    )
    proc_chain = {step_name: step_par}
    data_name = 'net_regime'
    
    # If the required data is not in the storage yet or recalculation is required
    if need_recalc or not dk.exists(data_name, proc_chain):
        
        # Read simulation result
        sim_res_desc = locate_sim_res_wc(sim_label, dirpath_res_hpc)
        with fs.open(sim_res_desc.fpath_pkl, 'rb') as fid:
            sim_result = pickle.load(fid)
    
        # Extract the regime from the sim result and store it locally
        R = sim_result['R']
        dk.store_data(R, data_name, proc_chain)
    
    # Load a previously stored regime
    R = dk.get_data(data_name, proc_chain)
    return R
    

