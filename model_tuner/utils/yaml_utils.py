from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Protocol

import dacite
from deepdiff import DeepDiff
import yaml


class DataclassProtocol(Protocol):
    """Protocol for dataclasses. """
    __dataclass_fields__: dict  # all dataclasses have this attribute

def _prepare_for_yaml(obj):
    """Prepare an object for saving to YAML. """
    if is_dataclass(obj):
        obj = asdict(obj)  # dataclass -> dict
    if isinstance(obj, tuple):
        return [_prepare_for_yaml(item) for item in obj]  # tuple -> list
    elif isinstance(obj, list):
        return [_prepare_for_yaml(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _prepare_for_yaml(value) for key, value in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value  # Enum -> str
    else:
        return obj

def save_yaml(obj: Any, fpath_yaml: str) -> None:
    """Save an object to YAML. """
    obj = _prepare_for_yaml(obj)
    with open(fpath_yaml, 'w') as fid:
        yaml.dump(obj, fid, default_flow_style=False)

def load_yaml(
        fpath_yaml,
        data_class: DataclassProtocol | None = None
        ) -> DataclassProtocol | dict:
    """Load an object from YAML. """
    with open(fpath_yaml, 'r') as fid:
        obj = yaml.safe_load(fid)
    if data_class is not None:
        if not is_dataclass(data_class):
            raise ValueError('data_class argument should be a dataclass')
        obj = dacite.from_dict(
            data_class=data_class,
            data=obj,
            config=dacite.Config(cast=[tuple, Enum])
        )
    return obj

def compare_yaml(obj1: Any, obj2: Any) -> bool:
    """Compare two objects converted to YAML. """
    obj1 = _prepare_for_yaml(obj1)
    obj2 = _prepare_for_yaml(obj2)
    return obj1 == obj2

def yaml_diff(obj1: Any, obj2: Any) -> DeepDiff:
    """Extract the difference two objects converted to YAML. """
    obj1 = _prepare_for_yaml(obj1)
    obj2 = _prepare_for_yaml(obj2)
    return DeepDiff(obj1, obj2)

