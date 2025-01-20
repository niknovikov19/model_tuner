@dataclass(frozen=True)
class ProcStepParams:
    pass

@dataclass(frozen=True)
class NetSpikesParams(ProcStepParams):
    pop_names: List[str] = None
    combine_cells: bool = True
    time_limits: list = (0, None)
    subtract_t0: bool = True  # make spike times relative to time_limits[0]
    ms: bool = False  # spike times in miliseconds, otherwise - in seconds
    ndigits: int = 6  # spike times are rounded up to this number of digits

class DataType(Enum):
    SIM_RESULT = 'sim_result'
    SPIKES = 'spikes'
    RATES = 'rates'

@dataclass
class GenericData:
    data_type: DataType = field(init=False)  # the default will be set in subclasses
    data_name: str | None = None
    params: ProcStepParams | None = None  # params used for producing this data
    src_params: Dict[str, ProcStepParams] = field(default_factory=dict) # params of the source data
    data: Any = None
    
    def __post_init__(self):
        if not self.data_name:
            self.data_name = self.data_type.value

@dataclass
class NetSpikesData(GenericData):
    data_type: DataType = field(init=False, default=DataType.SPIKES)
    params: NetSpikesParams | None = None
    data: Dict[str, List[np.ndarray]] = field(default_factory=dict)