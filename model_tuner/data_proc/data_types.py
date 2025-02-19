from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from .proc_params import ProcStepParams, ProcParamsChain
from .proc_params import NetSpikesParams, NetRatesParams


class DataType(Enum):
    #SIM_RESULT = 'sim_result'
    SPIKES = 'spikes'
    RATES = 'rates'


@dataclass(frozen=True)
class DataIndex:
    """Uniquely identifies a data entry stored in DataKeeper. """
    data_name: str
    params_chain: ProcParamsChain


@dataclass
class GenericData:
    data_type: DataType = field(init=False)  # the default will be set in subclasses
    data_name: str | None = None
    params: ProcStepParams | None = None  # params used for producing this data
    params_chain: ProcParamsChain = ProcParamsChain()  # params of the source data
    data: Any = None
    
    def __post_init__(self):
        if not self.data_name:
            self.data_name = self.data_type.value

# =============================================================================
# @dataclass
# class SimResultData(GenericData):
#     data_type: DataType = field(init=False, default=DataType.SIM_RESULT)
#     params: SimResultParseParams
#     data: Dict = field(default_factory=dict)
# =============================================================================

@dataclass
class NetSpikesData(GenericData):
    data_type: DataType = field(init=False, default=DataType.SPIKES)
    params: NetSpikesParams
    data: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    time_limits: Tuple[float, float] = (0, None)
    net_size: Dict[str, int] | None = None

@dataclass
class NetRatesData(GenericData):
    data_type: DataType = field(init=False, default=DataType.RATES)
    params: NetRatesParams
    data: Dict[str, List[float] | float] = field(default_factory=dict)
