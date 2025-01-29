from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import pickle
from pprint import pprint
from typing import Any, List, Dict

import numpy as np

from proc_params import ProcStepParams, NetSpikesParams, NetRatesParams
from data_types import DataType, NetSpikesData, NetRatesData

from sim_result import SimResultFile
from data_keeper import DataKeeper
from netpyne_batch_analyzer import BatchAnalyzer
from netpyne_result_parser import SimResultParserNetPyNE
from sim_data_proc import DataProcessor


class BatchMetricGetter(ABC):
    @abstractmethod
    def get_batch_par_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_batch_par_values(self, par_name: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_pop_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_pop_rates_batch(self, pop_name: str) -> np.ndarray:
        pass


class BatchMetricGetterSurrogate(BatchMetricGetter):
    
    def __init__(self):
        self._batch_params = {
            'rxe': np.linspace(0, 10, 20)
        }
        self._pop_names = ['pop1', 'pop2']
        self._gen_surrogate_data()
    
    def _gen_surrogate_data(self):
        self._rate_data = {}
        self._data_gen_par = {
            'pop1': {'a': 1.0, 'b': 0.1, 'k': 10.0, 's': 0.2},
            'pop2': {'a': 2.0, 'b': 0.2, 'k': 20.0, 's': 0.5}
        }
        for pop_name in self._pop_names:
            x = self._batch_params['rxe']
            a, b, k, s = [self._data_gen_par[pop_name][p]
                          for p in ('a', 'b', 'k', 's')]
            y = a * np.exp(b * x) + k + s * np.random.rand(len(x))
            self._rate_data[pop_name] = y
    
    def get_batch_par_names(self) -> List[str]:
        return list(self._batch_params.keys())
    
    def get_batch_par_values(self, par_name: str) -> np.ndarray:
        return self._batch_params[par_name]
    
    def get_pop_names(self) -> List[str]:
        return self._pop_names
    
    def get_pop_rates_batch(self, pop_name: str) -> np.ndarray:
        return self._rate_data[pop_name]


class BatchMetricGetter1D(BatchMetricGetter):
    
    def __init__(
            self,
            dirpath_batch: str | Path,
            exp_name: str,
            pop_names: List[str],
            batch_param_name: str,
            proc_step_params: Dict[str, ProcStepParams] = None
            ):
        
        self._pop_names = pop_names
        self._batch_param_values = {batch_param_name: []}        
        self._net_rates_list: List[NetRatesData] = []
        
        # Parameters of data processing steps
        self._proc_step_params = proc_step_params or {}
        if 'net_spikes' not in self._proc_step_params:
            self._proc_step_params['net_spikes'] = (
                NetSpikesParams(pop_names=self._pop_names)
            )
        if 'net_rates' not in self._proc_step_params:
            self._proc_step_params['net_rates'] = NetRatesParams()        
        
        # Get batch param values and calculate net_rates
        self._parse_batch_result(dirpath_batch, exp_name, batch_param_name)
    
    def get_batch_par_names(self) -> List[str]:
        return list(self._batch_params.keys())
    
    def get_batch_par_values(self, par_name: str) -> np.ndarray:
        return np.array(self._batch_param_values[par_name])
    
    def get_pop_names(self) -> List[str]:
        return self._pop_names
    
    def get_pop_rates_batch(self, pop_name: str) -> np.ndarray:
        return np.array(
            [net_rates.data[pop_name] for net_rates in self._net_rates_list]
        )
    
    def _parse_batch_result(
            self,
            dirpath_batch: str | Path,
            exp_name: str,
            batch_param_name: str
            ) -> None:
        """
        Parse sim results of several jobs from a batch and, based on them,
        fill self._batch_param_values[batch_param_name] and self._net_rates_list.
        
        """                
        # Initialize batch result analyzer
        ba = BatchAnalyzer(dirpath_batch, exp_name, par_names=[batch_param_name])
        
        # Get values of the batch parameter
        self._batch_param_values[batch_param_name] = (
            ba.get_all_jobs_param(batch_param_name)
        )
        
        # Initialize DataKeeper
        dirpath_batch = Path(dirpath_batch)
        dirpath_dk = str(dirpath_batch / 'data_keeper')
        os.makedirs(dirpath_dk, exist_ok=True)
        dk = DataKeeper(dirpath_dk)
        
        # Initialize sim result parser
        res_parser = SimResultParserNetPyNE(dk=dk)
        
        # Initialize data processor
        data_proc = DataProcessor(dk)
        
        self._net_rates_list = []
        
        for job_id in range(ba.num_jobs):
            
            fpath_sim_res = ba.get_job_data_path(job_id)
        
            # Open sim result for parsing
            sim_res_desc = SimResultFile(fpath_sim_res)
            res_parser.set_sim_result(result_desc=sim_res_desc, defer_load=True)
            
            # Extract spikes from the sim result and strore them into dk
            spikes_data_id = res_parser.extract_net_spikes(
                self._proc_step_params['net_spikes'],
                data_name_out=f'net_spikes_{job_id}'
            )
            
            # Load spikes from dk, calculate rates from them, save rates into dk
            rates_data_id = data_proc.calc_net_rates(
                spikes_data_id,
                self._proc_step_params['net_rates'],
                data_name_out=f'net_rates_{job_id}'
            )
            
            # Load rates from dk
            rates_dk = data_proc.load_data(rates_data_id)
            self._net_rates_list.append(rates_dk)
