from dataclasses import dataclass, is_dataclass, fields
from enum import Enum, auto
#from io import IOBase
import json
import os
from typing import Dict, List, Tuple, Union, Literal, Any

from fabric import Connection as FabricConnection
from fs.sshfs import SSHFS
import paramiko


class SSHClient:
    def __init__(self, host: str, user: str, port: int, fpath_private_key: str):
        self._host = host
        self._user = user
        self._port = port
        self._fpath_private_key = fpath_private_key
    
    def get_filesys_handler(self) -> SSHFS:
        private_key = paramiko.RSAKey(filename=self._fpath_private_key)
        return SSHFS(
            host=self._host,
            user=self._user,
            port=self._port,
            pkey=private_key
        )
    
    def get_conn_handler(self) -> FabricConnection:
        return FabricConnection(
            host=self._host,
            user=self._user,
            port=self._port,
            connect_kwargs={'key_filename': self._fpath_private_key}
        )
    
    def file_exists(self, fpath: str) -> bool:
        with self.get_filesys_handler() as fs:
            return fs.exists(fpath)