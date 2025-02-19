"""
Top-level class for extracting data from a NetPyNE simulation result.    
Based of high-level functions from data_proc_funcs module.
Reads inputs from DataKeeper object and stores results in it.

"""

import logging
import pickle

from .proc_params import ProcParamsChain
from .proc_params import SourceDataParams, NetSpikesParams
from .data_types import DataIndex

from .sim_result import SimResultFile
from . import data_proc_funcs as proc_funcs
from .data_keeper import DataKeeper


class SimResultParserNetPyNE:
    def __init__(
            self,
            dk: DataKeeper = None,
            result_desc: SimResultFile = None,
            defer_load: bool = True
            ):
        self.dk = None
        self.free()
        if dk:
            self.set_data_keeper(dk)
        if result_desc:
            self.set_sim_result(result_desc, defer_load=defer_load)
    
    def free(self) -> None:
        self.sim_res_data = None  # content
        self.sim_res_desc = None  # descriptor
        self.params_chain = None
    
    def set_data_keeper(self, dk: DataKeeper) -> None:
        """Set DataKeeper object responsible for storing parsed data. """
        self.dk = dk
    
    def _check(self) -> bool:
        if not self.dk:
            raise RuntimeError('DataKeeper was not assigned')
        if not self.sim_res_desc:
            raise RuntimeError('Simulation result was not assigned')
    
    def _load_data(self):
        self._check()
        if not self.sim_res_data:
            logging.debug('SimResultParserNetPyNE._load_data(): load')
            with self.sim_res_desc.open_file('rb') as fid:
                self.sim_res_data = pickle.load(fid)
        else:
            logging.debug('SimResultParserNetPyNE._load_data(): already loaded')

    def set_sim_result(
            self,
            result_desc: SimResultFile,
            force_reset: bool = False,
            defer_load: bool = True
            ) -> None:
        """Set the simulation result that will be used for parsing. """
        
        if (result_desc != self.sim_res_desc) or force_reset:
            logging.debug('SimResultParserNetPyNE.set_sim_result(): work with new data')
            self.free()
            self.sim_res_desc = result_desc
            self.params_chain = ProcParamsChain(
                {'source': SourceDataParams(filepath=result_desc.filepath)}
            )
        else:
            logging.debug('SimResultParserNetPyNE.set_sim_result(): re-use old data')
        
        # Load sim result data from a file or defer loading until the data is needed
        if not defer_load:
            logging.debug('SimResultParserNetPyNE.set_sim_result(): load data')
            self._load_data()
        else:
            logging.debug('SimResultParserNetPyNE.set_sim_result(): defer data loading')
    
    def extract_net_spikes(
            self,
            params: NetSpikesParams,
            data_name_out: str = 'net_spikes',
            recalc: bool = False
            ) -> DataIndex:
        
        logging.debug(
            f'SimResultParserNetPyNE.extract_net_spikes(): data = {data_name_out}'
        )
        self._check()
        
        # Generate output data index
        data_id_out = DataIndex(
            data_name_out,
            self.params_chain.add_step(data_name_out, params)
        )
        
        # Check if the step result already exists        
        if (self.dk.exists(data_id_out.data_name, data_id_out.params_chain)
                and not recalc):
            logging.debug(
                "SimResultParserNetPyNE.extract_net_spikes(): don't recalculate"
            )
            return data_id_out
        
        # Do a deferred data loading if needed
        self._load_data()
        
        # Perform the extraction
        data_out = proc_funcs.extract_net_spikes(
            sim_res_data=self.sim_res_data,
            par=params,
            params_chain_in=self.params_chain
        )
        
        # Store the result and return its index
        self.dk.store_data(
            data_out,
            data_name=data_id_out.data_name,
            data_params=data_id_out.params_chain
        )        
        return data_id_out
