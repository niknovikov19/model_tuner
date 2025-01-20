from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from io import IOBase
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

from fs.base import FS
#import numpy as np

#from data_keeper import DataKeeper
#from filesys import FileSystem, FileSystemLocal
#from model_tuner.opt.regimes import NetRegimeWC
#from proc_params import ProcStepParams


@dataclass(frozen=True)
class SimResult(ABC):
    
    @abstractmethod
    def exists(self) -> bool:
        pass
    
    @abstractmethod
    def delete(self) -> None:
        pass


@dataclass(frozen=True)
class SimResultFile(SimResult):
    fs: FS
    filepath: Path
    
    def exists(self) -> bool:
        return self.fs.exists(self.filepath)
    
    def open_file(self, *args, **kwargs) -> IOBase:
        if not self.exists():
            raise ValueError(f'Simulation result does not exist: {self.filepath}')
        return self.fs.open(self.filepath, *args, **kwargs)
    
    def delete(self) -> None:
        self.fs.remove(self.filepath)

# =============================================================================
#     def __eq__(self, other):
#         if type(other) is type(self):
#             return self.filepath == other.filepath
#         return False
#     
#     def __hash__(self):
#         return hash(self.filepath)
# =============================================================================

