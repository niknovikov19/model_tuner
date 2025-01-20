import pickle
from typing import Any

import netpyne_res_parse_utils as parse_utils
from proc_params import SourceDataParams, NetSpikesParams, ProcParamsChain
from data_types import DataType, NetSpikesData
from sim_result import SimResultFile


class SimResultParserNetPyNE:
    def __init__(self):
        self.sim_res_data = None
        self.sim_result = None
        self.params_chain = None

    def load_sim_result(
            self,
            result: SimResultFile,
            need_reload=False
            ):
        """Load simulation result from a pkl file. """
        if (result != self.sim_result) or need_reload:
            self.sim_result = result
            with result.open_file('rb') as fid:
                self.sim_res_data = pickle.load(fid)
            self.params_chain = {
                'source': SourceDataParams(filepath=result.filepath)
            }
    
    def free(self):
        self.sim_res_data = None
        self.sim_result = None
    
    def _check(self):
        if not self.sim_res_data or not self.sim_result:
            raise RuntimeError('Simulation result is not loaded')
    
    def extract_net_spikes(
            self,
            par: NetSpikesParams
            ) -> NetSpikesData:
        self._check()
        
        # Extract spikes from netpyne simulation result    
        S = parse_utils.get_net_spikes(
            self.sim_res,
            pop_names=par.pop_names,
            combine_cells=par.combine_cells,
            t0=par.time_limits[0],
            tmax=par.time_limits[1],
            subtract_t0=par.subtract_t0,
            ms=par.ms,
            ndigits=par.ndigits
        )
        
        # Get time limits of the spike data
        T = parse_utils.get_sim_duration(self.sim_res)
        t0 = par.time_limits[0]
        tmax = par.time_limits[1] or T
        if par.subtract_t0:
            tmax -= t0
            t0 = 0
        if par.ms:
            t0 *= 1000
            tmax *= 1000
        
        params_chain = self.params_chain.add_step('net_spikes', par)
        return NetSpikesData(
            data=S,
            time_limits=(t0, tmax),
            params=par,
            params_chain=params_chain
        )
