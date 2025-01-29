#import importlib
import json
import glob
#import logging
#import os
from pathlib import Path
#import pickle as pkl
from typing import Any, List, Dict


def _get_key_seq(x, seq):
    """Get x[seq[0]][seq[1]][...] """
    for key in seq:
        x = x[key]
    return x


class BatchAnalyzer:
    
    def __init__(
            self,
            dirpath_base: str | Path,
            exp_name: str,
            par_names: List[str]
            ):
        self.dirpath_base = Path(dirpath_base)
        self.par_names = par_names
        self.templ_cfg = f'{exp_name}_?????_cfg.json'
        self.templ_data = f'{exp_name}_?????_data.pkl'
        self.num_jobs = self._calc_num_jobs()
        self.par_vals_lst = []
        self._read_param_values()
    
    def _calc_num_jobs(self) -> int:
        templ = self.dirpath_base / self.templ_cfg
        return len(glob.glob(str(templ)))
    
    def get_job_cfg_path(self, job_id: int) -> str:
        fname = self.templ_cfg.replace('?????', f'{job_id:05d}')
        return str(self.dirpath_base / fname)
    
    def get_job_data_path(self, job_id: int) -> str:
        fname = self.templ_data.replace('?????', f'{job_id:05d}')
        return str(self.dirpath_base / fname)
    
    @classmethod
    def _extarct_par_from_cfg(cls, cfg: Dict, par_name: str) -> Any:
        """Get a dict field with nesting support (. -> []) """
        return _get_key_seq(cfg, par_name.split('.'))
        
    def _read_param_values(self) -> None:
        """Read parameter value combinations from batch job configs. """
        self.par_vals_lst = []
        for n in range(self.num_jobs):
            fpath_cfg = self.get_job_cfg_path(job_id=n)
            with open(fpath_cfg, 'r') as fid:
                cfg = json.load(fid)['simConfig']
            par_vals = {}
            for par_name in self.par_names:
                par_vals[par_name] = self._extarct_par_from_cfg(cfg, par_name)
            self.par_vals_lst.append(par_vals)
        
    def get_job_params(self, job_id: int) -> Dict[str, Any]:
        return self.par_vals_lst[job_id]
    
    def get_job_param(self, param_name: str, job_id: int) -> Any:
        return self.par_vals_lst[job_id][param_name]
    
    def get_all_jobs_param(self, param_name: str) -> List:
        return [self.get_job_param(param_name, job_id)
                for job_id in range(self.num_jobs)]
    
    