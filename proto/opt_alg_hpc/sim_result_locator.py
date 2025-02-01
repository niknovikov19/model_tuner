from pathlib import Path

from fs.base import FS

from sim_result import SimResultFile


class SimResultLocator:
    def __init__(self, dirpath_base: str | Path, fs: FS):
        fs.check()  # raises an exception if the filesystem is closed
        self.dirpath_base = Path(dirpath_base)
        self.fs = fs
    
    def _locate_result(self, sim_label: str) -> SimResultFile:
        fname_res = f'{sim_label}_data.pkl'
        fpath_res = self.dirpath_base / fname_res
        return SimResultFile(fs=self.fs, filepath=fpath_res)
    
    def result_exists(self, sim_label: str) -> bool:
        return self._locate_result(sim_label).exists()
    
    def locate_result(
            self,
            sim_label: str,
            should_exist: bool = True
            ) -> SimResultFile:
        res = self._locate_result(sim_label)
        if should_exist and not res.exists():
            raise RuntimeError(f'No result found for simulation {sim_label}')
        return res
