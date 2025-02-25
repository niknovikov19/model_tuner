
from model_tuner.opt.uc_mappers import NetUCMapper1D
from .uc_map_config import UCMapFitParams


def init_uc_mapper(par: UCMapFitParams) -> NetUCMapper1D:
    """Initialize unconnected-to-connected regime mapper. """    
    uc_mapper = NetUCMapper1D(
        pop_names=par.pop_names,
        map_type=par.map_type,
        map_params=par.map_params
    )
    uc_mapper.set_to_identity()