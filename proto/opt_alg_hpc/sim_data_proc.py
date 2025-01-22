from dataclasses import dataclass, field
import logging
from typing import Any, Dict
from typing import TypeVar, Callable

import numpy as np

from proc_params import ProcStepParams, ProcParamsChain
from proc_params import NetSpikesParams, NetRatesParams
from data_types import GenericData, DataIndex
#from data_types import NetSpikesData, NetRatesData

import data_proc_funcs as proc_funcs
from data_keeper import DataKeeper


TDataInp = TypeVar('TDataInp', bound=GenericData)
TDataOut = TypeVar('TDataOut', bound=GenericData)
TProcParams = TypeVar('TProcParams', bound=ProcStepParams)


class DataProcessor:
    def __init__(self, dk: DataKeeper = None):
        self.set_data_keeper(dk)
    
    def set_data_keeper(self, dk: DataKeeper) -> None:
        """Set DataKeeper object responsible for storing parsed data. """
        self.dk = dk
    
    def _check(self):
        if not self.dk:
            raise RuntimeError('DataKeeper was not assigned')
    
    def load_data(self, data_id: DataIndex, **kwargs) -> GenericData:
        return self.dk.get_data(
            data_name=data_id.data_name,
            data_params=data_id.params_chain,
            **kwargs
        )
    
    def _proc_step(
            self,
            data_id_in: DataIndex,
            step_func: Callable[[TDataInp, TProcParams, str | None], TDataOut],
            step_params: ProcStepParams,
            data_name_out: str,
            recalc: bool = False
            ) -> DataIndex:
        
        self._check()

        # Generate output data index
        data_id_out = DataIndex(
            data_name_out,
            data_id_in.params_chain.add_step(data_name_out, step_params)
        )
        
        # Check if the step result already exists        
        if (self.dk.exists(data_id_out.data_name, data_id_out.params_chain)
            and not recalc):
            return data_id_out
        
        # Load input data
        data_in = self.dk.get_data(
            data_id_in.data_name,
            data_id_in.params_chain
        )
        
        # Perform the processing step
        data_out = step_func(data_in, step_params, data_name_out)
        
        # Store the result and return its index
        self.dk.store_data(
            data_out,
            data_name=data_id_out.data_name,
            data_params=data_id_out.params_chain
        )        
        return data_id_out
    
    def calc_net_rates(
            self,
            data_id_in: DataIndex,
            step_params: NetRatesParams,
            data_name_out: str = 'net_rates',
            recalc: bool = False
            ) -> DataIndex:
        """Calculate pop. firing rates from spike trains. """
        logging.debug('DataProcessor.calc_net_rates()')
        return self._proc_step(
            data_id_in,
            step_func=proc_funcs.calc_net_rates,
            step_params=step_params,
            data_name_out=data_name_out,
            recalc=recalc
        )
    