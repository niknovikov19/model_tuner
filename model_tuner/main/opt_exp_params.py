from dataclasses import dataclass, field
from typing import Any, Tuple, Dict
import numpy as np


@dataclass
class OptExperimentParams:
    
    # Experiment name
    exp_name: str = 'OPT_EXP'

    # Batchtools script to run
    fpath_batch_script_hpc: str = ''

    # HPC base foder for the experiments data (exp. name will be added)
    dirpath_hpc_base: str = ''

    # Conda environment to use on HPC
    conda_env: str = ''
    
    # Names of the populations
    pop_names: Tuple[str, ...] = ()

    # Original target regime (firing rates of the populations)
    rr_base: Dict[str, float] = field(default_factory=dict)

    # Target regime multipliers used in the algorithm
    pfr_vec: np.ndarray = field(default_factory=lambda: np.array([]))

    # Rate of U-C mapping change between iterations (0 = old, 1 = replace)
    uc_alpha: float = 0.25

    # Global weight multiplier
    # keep it outside model_cfg for compatibility
    #wmult: float = 0.25
    
    # Parameters to override in the cfg of the model
    model_cfg: Dict[str, Any] = field(default_factory=dict)