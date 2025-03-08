from pathlib import Path

from fs.base import FS

from model_tuner.data_proc import SimResultFile


class SimResultLocator:
    """
    Locates simulation result in a file system by sim_label.

    """

    def __init__(
            self,
            dirpath_base: str | Path,
            fs: FS,
            filename_templ: str = '{sim_label}_data.pkl'
            ):
        fs.check()  # raises an exception if the filesystem is closed
        self.dirpath_base = Path(dirpath_base)
        self.fs = fs
        self.filename_templ = filename_templ
    
    def _locate_result(self, sim_label: str) -> SimResultFile:
        fname_res = self.filename_templ.format(sim_label=sim_label)
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
