from abc import ABC
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimResultDesc(ABC):
    pass

@dataclass(frozen=True)
class SimResultDescPKL(SimResultDesc):
    fpath_pkl: Path