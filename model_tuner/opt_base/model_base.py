from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ModelDesc(ABC):
    @abstractmethod
    def get_pop_names(self) -> List[str]: pass