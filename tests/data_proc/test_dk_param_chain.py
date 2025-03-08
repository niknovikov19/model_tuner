"""
Test storing of processing steps with associated
parameter chains into DataKeeper.

"""

from dataclasses import dataclass
from copy import deepcopy
import os
from pprint import pprint
import shutil

from model_tuner.data_proc import (
    ProcStepParams,
    NetSpikesParams,
    DataKeeper
)


@dataclass(frozen=True)
class NestedParams1(ProcStepParams):
    x1: int = 1
    
@dataclass(frozen=True)
class NestedParams2(ProcStepParams):
    x2: int = 2
    par2: NestedParams1 = NestedParams1()

@dataclass(frozen=True)
class NestedParams3(ProcStepParams):
    x3: int = 3
    par3: NestedParams2 = NestedParams2()


dirpath_dk = r'D:\WORK\Salvador\repo\model_tuner\test_data\test_dk_param_chain'
if os.path.exists(dirpath_dk):
    shutil.rmtree(dirpath_dk, ignore_errors=True)
os.makedirs(dirpath_dk, exist_ok=True)

dk = DataKeeper(dirpath_dk)

data_info = {}

data_info['sim_result'] = {
    'step_params': None,
    'data': 'SIM_RESULT_DATA'
}
data_info['spikes'] = {
    'step_params': NetSpikesParams(),
    'data': 'SPIKE_DATA'
}
data_info['proc_step'] = {
    'step_params': ProcStepParams(),
    'data': 'PROC_STEP_DATA'
}
data_info['proc_step_2'] = {
    'step_params': NestedParams3(),
    'data': 'PROC_STEP_DATA_2'
}

param_chain = {}
for data_name, info in data_info.items():
    param_chain[data_name] = info['step_params']
    dk.store_data(info['data'], data_name, deepcopy(param_chain))

# Print metadata
pprint(dk.list_data())
