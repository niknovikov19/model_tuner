from dataclasses import fields, is_dataclass
from enum import Enum, auto
import hashlib
import json
import os
from pathlib import Path
import pickle as pkl
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr


class CustomEncoder(json.JSONEncoder):
    """JSON encoder that treats ndarrays and dataclasses. """
    def treat_dataclass(self, obj):
        return obj.__dict__    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            obj = int(obj)
        if is_dataclass(obj):
            obj = self.treat_dataclass(obj)
        return json.JSONEncoder.encode(self, obj)
    
class NonDefEncoder(CustomEncoder):
    """JSON encoder: remove dataclass fields with default values. """
    def treat_dataclass(self, obj):
        return {field.name: getattr(obj, field.name) for field in fields(obj)
                if getattr(obj, field.name) != field.default}


class DataFormat(Enum):
    PKL = auto()
    JSON = auto()
    XR = auto()
    
    def get_extention(self) -> str:
        extensions = {
            DataFormat.PKL: '.pkl',
            DataFormat.JSON: '.json',
            DataFormat.XR: '.nc'
        }
        return extensions[self]
    
    @classmethod
    def from_extension(cls, extension: str) -> 'DataFormat':
        """Convert a file extension string to the corresponding DataFormat."""
        extension_map = {
            '.pkl': cls.PKL,
            '.json': cls.JSON,
            '.nc': cls.XR
        }
        if extension not in extension_map:
            raise ValueError(f"Unknown extension: {extension}")
        return extension_map[extension]

def _load_formatted_data(fpath_data: str, **kwargs) -> Any:
    """Load data from a file using a method based on the file extension. """
    
    data_format = DataFormat.from_extension(Path(fpath_data).suffix)
    
    if data_format == DataFormat.PKL:
        with open(fpath_data, 'rb') as fid:
            data = pkl.load(fid)
        return data
    
    elif data_format == DataFormat.JSON:
        with open(fpath_data, 'r') as fid:
            data = json.load(fid, cls=CustomEncoder)
        return data
    
    elif data_format == DataFormat.XR:
        return data.to_netcdf(fpath_data, engine='h5netcdf', **kwargs)
    
    else:
        raise ValueError(f'Unsupported data format: {data_format}')

def _save_formatted_data(data: Any, fpath_data: str, **kwargs) -> None:
    """Save data to a file using a method based on the file extension. """
    
    data_format = DataFormat.from_extension(Path(fpath_data).suffix)
    
    dirpath_data = str(Path(fpath_data).parent)
    os.makedirs(dirpath_data, exist_ok=True)
    
    if data_format == DataFormat.PKL:
        with open(fpath_data, 'wb') as fid:
            pkl.dump(data, fid)
    
    elif data_format == DataFormat.JSON:
        with open(fpath_data, 'w') as fid:
            json.dump(data, fid, cls=CustomEncoder)
    
    elif data_format == DataFormat.XR:
        return xr.open_dataarray(fpath_data, **kwargs)
    
    else:
        raise ValueError(f'Unsupported data format: {data_format}')


class DataKeeper:
    def __init__(
            self,
            storage_dir: str,
            metadata_file: str = 'metadata.json'):
        # Directory where data will be stored
        self.storage_dir = Path(storage_dir)
        # Metadata to track stored data
        self.metadata_file = self.storage_dir / metadata_file
        self.data_index = self._load_metadata()
        
    def _hash_params(self, params: Any, exclude_default=False) -> str:
        """Create a hash from params to generate a unique key for each data entry. """
        if exclude_default:
            # Don't hash dataclass fields with default values (for backward compatibility)
            params_str = json.dumps(params, sort_keys=True, cls=NonDefEncoder)
        else:
            params_str = json.dumps(params, sort_keys=True, cls=CustomEncoder)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _generate_data_key(self, name: str, params: Any) -> str:
        """Generate a unique key for a data entry based on its name and params. """
        return f'{name}_{self._hash_params(params)}'
    
    def _generate_data_path_rel(
            self,
            data_name: str,
            data_params: Any,
            data_format: DataFormat = DataFormat.PKL
            ) -> str:
        """Generate filepath to store data entry (relative to self.storage_dir). """
        data_key = self._generate_data_key(data_name, data_params)
        fname = data_key + data_format.get_extention()
        return str(Path(data_name) / fname)
        
    def _generate_data_path_abs(
            self,
            data_name: str,
            data_params: Any,
            data_format: DataFormat = DataFormat.PKL
            ) -> str:
        """Generate filepath to store data entry (absolute). """
        fpath_rel = self._generate_data_path_rel(
            data_name, data_params, data_format
        )
        return str(self.storage_dir / fpath_rel)

    def _load_metadata(self) -> Dict:
        """Load the metadata file that contains information about stored data entries. """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata about stored data entries in a human-readable format (JSON). """
        with open(self.metadata_file, 'w') as f:
            json.dump(self.data_index, f, indent=4, cls=CustomEncoder)
            
    def exists(
            self,
            data_name: str,
            data_params: Any
            ) -> bool:
        """Check if a data entry exists. """
        key = self._generate_data_key(data_name, data_params)
        return (key in self.data_index)
    
    def get_data(
            self,
            data_name: str,
            data_params: Any,
            **kwargs
            ) -> Any:
        """Load data entry. """
        data_params = data_params or {}
        if self.exists(data_name, data_params):
            key = self._generate_data_key(data_name, data_params)
            fpath_rel = self.data_index[key]['filepath_rel']
            fpath_abs = str(self.storage_dir / fpath_rel)
            return _load_formatted_data(fpath_abs, **kwargs)
        else:
            raise ValueError(f'Data "{data_name}" not found')

    def store_data(
            self,
            data: Any,
            data_name: str,
            data_params: Any,
            data_format: DataFormat = DataFormat.PKL,
            allow_rewrite: bool = False,
            **kwargs
            ) -> None:
        """Store data to disk. """
        
        data_params = data_params or {}        
        if self.exists(data_name, data_params) and not allow_rewrite:
            raise RuntimeError('Data rewriting is prohibited')
         
        # Add an entry to metadata
        key = self._generate_data_key(data_name, data_params)
        fpath_data_rel = self._generate_data_path_rel(
            data_name, data_params, data_format
        )
        self.data_index[key] = {
            'name': data_name,
            'params': data_params,
            'filepath_rel': fpath_data_rel
        }
        self._save_metadata()
        
        # Save the data
        fpath_data_abs = self._generate_data_path_abs(
            data_name, data_params, data_format
        )
        _save_formatted_data(data, fpath_data_abs, **kwargs)

    def list_data(self):
        """Return a human-readable format of stored data information."""
        return self.data_index
    
    # TODO: implement chunking
    # Params list insteead of dict?
    # A custom decorator for json serialization of dataclass?
