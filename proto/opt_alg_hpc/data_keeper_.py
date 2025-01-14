from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr

from proc_params import ProcStepParams


class DataKeeper:

    def exists(
            self,
            data_name: str,
            proc_params: Dict[str, ProcStepParams | Any]
            ) -> bool:
        """Check if a data entry exists. """
        pass

    def get_data(
            self,
            data_name: str,
            proc_params: Dict[str, ProcStepParams | Any] = None,
            **kwargs
            ) -> Any:
        """Load data entry. """
        proc_params = proc_params or {}
        if self.exists(data_name, proc_params):
            return None
        else:
            raise ValueError(f'Data "{data_name}" not found')

    def store_data(
            self,
            data: Any,
            data_name: str,
            proc_params: Dict[str, ProcStepParams | Any] = None,
            allow_rewrite=True,
            **kwargs
            ):
        """Store data entry. """
        proc_params = proc_params or {}
        if self.exists(data_name, proc_params) and not allow_rewrite:
            raise ValueError('Data rewriting is prohibited')
        pass
