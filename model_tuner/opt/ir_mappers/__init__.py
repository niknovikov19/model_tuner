from .ir_mapper_base import PopIRMapper, NetIRMapper
from .ir_mapper_emp_1d import PopIREmpiricalMapper1D, NetIREmpiricalMapper1D
from .ir_mapper_emp_1d_slice import PopIRMapper1DSlice, NetIRMapper1DSlice
from .ir_mapper_wc import PopIRMapperWC, NetIRMapperWC

__all__ = [
    'PopIRMapper',
    'NetIRMapper',
    'PopIREmpiricalMapper1D',
    'NetIREmpiricalMapper1D',
    'PopIRMapper1DSlice',
    'NetIRMapper1DSlice',
    'PopIRMapperWC',
    'NetIRMapperWC'
]

