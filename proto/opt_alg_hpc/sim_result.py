from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from io import IOBase
import os
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
    filepath: Path
    fs: FS | None = None
    
    def exists(self) -> bool:
        if self.fs:
            return self.fs.exists(self.filepath.as_posix())
        else:
            return os.path.exists(str(self.filepath))
    
    def open_file(self, *args, **kwargs) -> IOBase:
        if not self.exists():
            raise ValueError(f'Simulation result does not exist: {self.filepath}')
        if self.fs:
            return self.fs.open(self.filepath.as_posix(), *args, **kwargs)
        else:
            return open(str(self.filepath), *args, **kwargs)
    
    def delete(self) -> None:
        if self.fs:
            self.fs.remove(self.filepath.as_posix())
        else:
            os.remove(str(self.filepath))

# =============================================================================
#     def __eq__(self, other):
#         if type(other) is type(self):
#             return self.filepath == other.filepath
#         return False
#     
#     def __hash__(self):
#         return hash(self.filepath)
# =============================================================================

