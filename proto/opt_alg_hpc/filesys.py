from abc import ABC, abstractmethod
from io import IOBase
from pathlib import Path


class FileSystem(ABC):
    @abstractmethod
    def open(fpath, mode) -> IOBase:
        pass

class FileSystemLocal(FileSystem):
    def open(fpath, mode) -> IOBase:
        return open(fpath, mode)

class FileSystemSSHCached(FileSystem):
    """
    Reading: download a file from SSHFS an open the local copy
    Contains mapping between local and ssh-based fs
    """
    def open(fpath, mode) -> IOBase:
        pass