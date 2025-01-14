from dataclasses import dataclass


@dataclass(frozen=True)
class ProcStepParams:
    pass

@dataclass(frozen=True)
class SpikeTrainParams(ProcStepParams):    
    combine_cells: bool = True
    time_limits: list = (0, None)
    subtract_t0: bool = True  # make spike times relative to time_limits[0]
    ms: bool = False  # spike times in miliseconds, otherwise - in seconds
    ndigits: int = 6  # spike times are rounded up to this number of digits 