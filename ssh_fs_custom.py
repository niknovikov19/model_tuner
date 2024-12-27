from pathlib import Path

from fs.copy import copy_file
from fs.osfs import OSFS
from fs.sshfs import SSHFS
import paramiko

from ssh_params import SSHParams


class SSHFSCustom(SSHFS):
    def __init__(
            self,
            ssh_par: SSHParams
            ):
        if ssh_par.fpath_private_key is not None:
            private_key = paramiko.RSAKey(filename=ssh_par.fpath_private_key)
        else:
            private_key = None
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