import pickle
from typing import Any

from filesys import FileSystem, FileSystemLocal
import netpyne_res_parse_utils as parse_utils
from proc_params import SpikeTrainParams
from sim_res_desc import SimResultDescPKL


class SimResultParserNP:
    def __init__(self):
        self.sim_res = None
        self.sim_res_desc = None

    def load_sim_result(
            self,
            sim_res_desc: SimResultDescPKL,
            fs: FileSystem = FileSystemLocal(),
            need_reload=False
            ):
        """Load simulation result from pkl file. """
        if (sim_res_desc != self.sim_res_desc) or need_reload:
            self.sim_res_desc = sim_res_desc
            with fs.open(sim_res_desc.fpath_pkl, 'rb') as fid:
                self.sim_res = pickle.load(fid)
    
    def extract_net_spikes(
            self,
            sim_res_desc: SimResultDescPKL,
            par: SpikeTrainParams
            ) -> Any:
        self.load_sim_result(sim_res_desc)
        return parse_utils.get_pop_spikes(
            self.sim_res, par.pop_name, combine_cells=False,
            t0=par.time_limits[0], tmax=par.time_limits[1],
            subtract_t0=par.subtract_t0, ms=par.ms, ndigits=par.ndigits
        )