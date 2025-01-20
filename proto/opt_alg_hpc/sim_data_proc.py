from dataclasses import dataclass, field
from typing import Any, Dict
from typing import TypeVar, Callable

import numpy as np

from proc_params import ProcStepParams, ProcParamsChain
from data_types import DataType, GenericData

from proc_params import NetSpikesParams, NetRatesParams
from data_types import NetSpikesData, NetRatesData

import data_proc_funcs as proc_funcs

from data_keeper import DataKeeper


TDataInp = TypeVar('TDataInp', bound=GenericData)
TDataOut = TypeVar('TDataOut', bound=GenericData)
TProcParams = TypeVar('TProcParams', bound=ProcStepParams)


@dataclass(frozen=True)
class DataIndex:
    """Uniquely identifies a data entry stored in DataKeeper. """
    data_name: str
    params_chain: ProcParamsChain


class DataProcessor:
    def __init__(self, dk: DataKeeper = None):
        self.set_data_keeper(dk)
    
    def set_data_keeper(self, dk: DataKeeper):
        """Set DataKeeper object responsible for storing parsed data. """
        self.dk = dk
    
    def _check(self):
        if self.dk is None:
            raise ValueError('DataKeeper was not assigned')
    
    @classmethod
    def _gen_data_id_out(
            cls,
            data_id_in: DataIndex,
            step_params: ProcStepParams,
            data_name_out: str
            ) -> DataIndex:
        return DataIndex(
            data_name_out,
            data_id_in.params_chain.add_step(data_name_out, step_params)
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
        data_id_out = self._gen_data_id_out(
            data_id_in=data_id_in,
            step_params=step_params,
            data_name_out=data_name_out            
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
        data_out = step_func(data_in, data_id_in.params_chain, data_name_out)
        
        # Store the result and return its index
        self.dk.store_data(
            data_out,
            data_name=data_id_out.data_name,
            data_params=data_id_out.params_chain
        )        
        return data_id_out
    
    def calc_net_rates(
            self,
            data_name_in: str,
            step_params: ProcStepParams,
            data_name_out: str = None,
            recalc: bool = False
            ) -> DataIndex:
        """Calculate pop. firing rates from spike trains. """
        data_name_out = data_name_out or 'net_rates'
        
        
        self._check()
        out_name = out_name or 'pop_rates'
        
        out_name, out_params = self._gen_out_name_par(
            inp_name, inp_params, out_name, step_name='bip')
        if recalc or not self.dk.exists(out_name, out_params):       
            with self.dk.get_data(inp_name, inp_params) as X:
                Y = xr_proc.calc_xr_diff(X, n=1)            
                self.dk.store_data(Y, out_name, out_params)
        return out_name, out_params
    
    def _calc_pop_rate(self, spikes, rate_par: RateParams):
        T = rate_par.time_limits[1] - rate_par.time_limits[0]  # TODO: account for tmax=None
        nspikes = np.array([len(s) for s in spikes])
        rr = nspikes / T
        return rr
    