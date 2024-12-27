from dataclasses import dataclass, is_dataclass, fields
from typing import Dict, List, Tuple, Union, Literal, Any, Optional

from fabric import Connection
#import paramiko


@dataclass
class SSHParams:
    host: str
    user: str
    port: Optional[int] = 22
    fpath_private_key: Optional[str] = None


class SSHConnCustomCloseError(Exception):
    """Raised when one or more errors occur while closing an SSH chain."""
    def __init__(self, message: str, inner_exceptions: list[Exception]):
        super().__init__(message)
        self.inner_exceptions = inner_exceptions


class SSHConnCustom(Connection):
    def __init__(self, ssh_par: List[SSHParams]):
        # Create intermediate ssh connections
        conn_chain = self._create_proxy_chain(ssh_par[:-1])
        gateway = conn_chain[-1] if conn_chain else None
        # Create the final ssh connection 9self)
        super().__init__(
            host=ssh_par[-1].host,
            user=ssh_par[-1].user,
            port=ssh_par[-1].port,
            connect_kwargs={"key_filename": ssh_par[-1].fpath_private_key},
            gateway=gateway
        )
        print(f'Opened: {self}')
        conn_chain.append(self)
        self._chain_connections = conn_chain
    
    @staticmethod
    def _create_proxy_chain(ssh_par_list: List[SSHParams]) -> List[Connection]:
        proxy_chain = []
        gateway = None
        for idx, par in enumerate(ssh_par_list):
            conn = Connection(
                host=par.host,
                user=par.user,
                port=par.port,
                connect_kwargs={"key_filename": par.fpath_private_key},
                gateway=gateway
            )
            print(f'Opened: {conn}')
            proxy_chain.append(conn)
            gateway = conn
        return proxy_chain
    
    @staticmethod
    def _close_proxy_chain(conn_chain: List[Connection]) -> None:
        # Close intermediate connections in reverse order
        exceptions = []
        for conn in reversed(conn_chain):
            try:
                if conn.is_connected:
                    print(f'Close: {conn}')
                    conn.close()
            except Exception as e:
                exceptions.append(e)
        if exceptions:
            raise SSHConnCustomCloseError(
                "Errors occurred while closing proxy connections",
                exceptions
            )
    
    def close(self):
        exceptions = []
        # Close the final connection (self)
        try:
            if self.is_connected:
                print(f'Close: {self}')
                super().close()
        except Exception as e:
            exceptions.append(e)
        
        # Close intermediate connections in reverse order
        try:
            self._close_proxy_chain(self._chain_connections[:-1])
        except Exception as e:
            exceptions.append(e)
        
        self._chain_connections = []
        
        if exceptions:
            raise SSHConnCustomCloseError(
                "Errors occurred while closing connections in SSHConnCustom.",
                exceptions
            )
        

# SSH parameters
ssh_par_lethe = SSHParams(
    host='lethe.downstate.edu',
    user='niknovikov19',
    port=1415,
    fpath_private_key=r'C:\Users\aleks\.ssh\id_rsa_lethe'
)
ssh_par_grid = SSHParams(
    host='grid',
    user='niknovikov19',
    fpath_private_key=r'C:\Users\aleks\.ssh\id_ed25519_grid'
)

#conn = SSHConnCustom([ssh_par_lethe])
conn = SSHConnCustom([ssh_par_lethe, ssh_par_grid])
result = conn.run('uname -a', hide=True)
print(result.stdout.strip())
conn.close()

