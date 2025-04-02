import importlib.util
from pathlib import Path
import os
import sys


def load_module(fpath_mod: str | Path):
    if isinstance(fpath_mod, Path):
        fpath_mod = fpath_mod.as_posix()
    if not os.path.isfile(fpath_mod):
        raise FileNotFoundError(f"File {fpath_mod} does not exist")
    mod_spec = importlib.util.spec_from_file_location(
        'module.name', fpath_mod)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules['module.name'] = mod
    mod_spec.loader.exec_module(mod)
    return mod