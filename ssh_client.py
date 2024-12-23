from dataclasses import dataclass, is_dataclass, fields
from enum import Enum, auto
#from io import IOBase
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal, Any

from fabric import Connection as FabricConnection
from fs.copy import copy_file
from fs.osfs import OSFS
from fs.sshfs import SSHFS
import paramiko


@dataclass
class SSHParams:
    host: str
    user: str
    port: int
    fpath_private_key: str

class SSHFSCustom(SSHFS):
    def __init__(
            self,
            ssh_par: SSHParams
            ):
        private_key = paramiko.RSAKey(filename=ssh_par.fpath_private_key)
        super().__init__(
            host=ssh_par.host,
            user=ssh_par.user,
            port=ssh_par.port,
            pkey=private_key
        )
        self._ssh_par = ssh_par
    
    def upload_file(self, fpath_local: str, fpath_ssh: str) -> None:
        fpath_local = Path(fpath_local)
        root_local = fpath_local.drive
        fpath_local_rel = fpath_local.relative_to(root_local).as_posix()
        with OSFS(root_local) as fs_local:
            copy_file(fs_local, fpath_local_rel, self, fpath_ssh)

class SSHConnCustom(FabricConnection):
    def __init__(
            self,
            ssh_par: SSHParams
            ):
        super().__init__(
            host=ssh_par.host,
            user=ssh_par.user,
            port=ssh_par.port,
            connect_kwargs={'key_filename': ssh_par.fpath_private_key}
        )

class SSHClientCloseError(Exception):
    """Raised when one or more errors occur while closing SSHClient resources."""
    def __init__(self, message: str, inner_exceptions: list[Exception]):
        super().__init__(message)
        self.inner_exceptions = inner_exceptions

class SSHClient:
    def __init__(self, ssh_par: SSHParams):
        self._ssh_par = ssh_par
        self.fs = SSHFSCustom(ssh_par)
        self.conn = SSHConnCustom(ssh_par)
        self._is_open = True

    def close(self):
        if not self._is_open:
            return    
        exceptions = []    
        # Close the filesystem
        try:
            self.fs.close()
        except Exception as e:
            exceptions.append(e)    
        # Close the Fabric connection
        try:
            self.conn.close()
        except Exception as e:
            exceptions.append(e)    
        self._is_open = False    
        if exceptions:
            raise SSHClientCloseError("One or more errors occurred while closing SSHClient resources.", exceptions)

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exiting the 'with' block, close resources."""
        self.close()
        