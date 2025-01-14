from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from io import IOBase
import pickle as pkl
from typing import Any, Dict, List, Optional

import numpy as np


class SimDataType(Enum):
    SPIKES = auto()
    RATE = auto()


class FileSystem(ABC):
    @abstractmethod
    def open_file(fpath, mode) -> IOBase:
        pass

class FileSystemLocal(FileSystem):
    def open_file(fpath, mode) -> IOBase:
        return open(fpath, mode)

class FileSystemSSHCached(FileSystem):
    """
    Reading: download a file from SSHFS an open the local copy
    Contains mapping between local and ssh-based fs
    """
    def open_file(fpath, mode) -> IOBase:
        pass


@dataclass
class SimData(ABC):
    pass

@dataclass
class SimDataRate(SimData):
    rate: float

@dataclass
class SimDataSpikes(SimData):
    spikes: List[float] = None


@dataclass
class SimResultDesc(ABC):
    pass

@dataclass
class SimResultDescWC(SimResultDesc):
    fpath_result: str = ''

@dataclass
class SimResultDescNetPyNE(SimResultDesc):
    fpath_result: str = ''
    fpath_spikes: Optional[str] = None


class SimResultParser:    
    def __init__(self, fs: FileSystem = FileSystemLocal()):
        self._fs = fs
    
    def get_sim_data(
            self,
            sim_desc: SimResultDesc,
            data_type: SimDataType,
            #data_params: Dict[str, Any] = None
            ) -> SimData:
        if data_type == SimDataType.RATE:
            return self.get_rate(sim_desc)
        elif data_type == SimDataType.SPIKES:
            return self.get_spikes(sim_desc)
        else:
            raise ValueError('Unsupported data type: {data_type}')
    
    def get_rate(
            self,
            sim_desc: SimResultDesc,
            #data_params: Dict[str, Any] = None
            ) -> SimData:
        raise RuntimeError('Not implemented: get_rate()')
    
    def get_spikes(
            self,
            sim_desc: SimResultDesc,
            #data_params: Dict[str, Any] = None
            ) -> SimData:
        raise RuntimeError('Not implemented: get_spikes()')

class SimResultParserWC(SimResultParser):    
    def get_rate(self, sim_desc: SimResultDescWC) -> SimDataRate:
        with self._fs.open(sim_desc.fpath_result, 'rb') as fid:
            res = pkl.load(fid)
        return SimDataRate(rate=res['rate'])

class SimResultParserNetPyNE(SimResultParser):
    def get_spikes(self, sim_desc: SimResultDescNetPyNE) -> SimDataSpikes:
        if sim_desc.fpath_spikes:
            with self._fs.open(sim_desc.fpath_spikes, 'rb') as fid:
                res = pkl.load(fid)
            return SimDataSpikes(spikes=res['spikes'])
        else:
            with self._fs.open(sim_desc.fpath_result, 'rb') as fid:
                res = pkl.load(fid)
            return SimDataSpikes(spikes=res['sim_data']['spikes'])
        
            




