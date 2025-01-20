#import numpy as np

from proc_params import NetSpikesParams, NetRatesParams
from data_types import DataType, NetSpikesData, NetRatesData

import data_proc_utils as proc_utils


def calc_net_rates(
        net_spikes: NetSpikesData,
        par: NetRatesParams,
        data_name: str | None = None
        ) -> NetRatesData:
    time_limits = par.time_limits
    time_limits[1] = time_limits[1] or net_spikes.time_limits[1]  # resolve None
    net_rates = proc_utils.calc_net_rates(
        net_spikes.data, time_limits, par.pop_names
    )
    data_name = data_name or 'net_rates'
    par_chain = net_spikes.params_chain.add_step(data_name, par)
    return NetRatesData(
        data=net_rates,
        data_type=DataType.RATES, data_name=data_name,
        params=par, params_chain = par_chain
    )