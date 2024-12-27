from typing import List

from fabric import Connection

from ssh_params import SSHParams


class SSHConnCustomCloseError(Exception):
    """Raised when one or more errors occur while closing an SSH chain. """
    def __init__(self, message: str, inner_exceptions: list[Exception]):
        super().__init__(message)
        self.inner_exceptions = inner_exceptions


class SSHConnCustom(Connection):
    """Custom SSH connection class that supports multi-hop SSH tunneling.
    
    This class extends the functionality of fabric.Connection by creating
    and managing intermediate SSH connections (proxies) to allow access
    to remote hosts through multiple gateways. 
    """
    def __init__(self, ssh_par: SSHParams | List[SSHParams]):
        if isinstance(ssh_par, SSHParams):
            ssh_par = [ssh_par]
        # Create intermediate ssh connections
        conn_chain = self._create_proxy_chain(ssh_par[:-1])
        gateway = conn_chain[-1] if conn_chain else None
        # Create the final ssh connection (self)
        # NOTE: all keys should be in the local filesystem
        ssh_par_final = ssh_par[-1]
        super().__init__(
            host=ssh_par_final.host,
            user=ssh_par_final.user,
            port=ssh_par_final.port,
            connect_kwargs={"key_filename": ssh_par_final.fpath_private_key},
            gateway=gateway
        )
        print(f'Opened: {self}')
        conn_chain.append(self)
        # Store conn_chain into a property
        # Defining a property before super().__init__() causes stack ovderflow
        self._chain_connections = conn_chain
    
    @staticmethod
    def _create_proxy_chain(ssh_par_list: List[SSHParams]) -> List[Connection]:
        """Create intermediate ssh connections (if needed). """
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
        """Close intermediate connections (if any). """
        exceptions = []
        for conn in reversed(conn_chain):  # close in reverse order
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
        """Close all ssh connections (self and proxies). """
        exceptions = []
        
        # Close the final connection (self)
        try:
            if self.is_connected:
                print(f'Close: {self}')
                super().close()
        except Exception as e:
            exceptions.append(e)
        
        # Close intermediate connections
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
    
    # Don't re-define __enter__(), as all the job is done by __init__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
            super().__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            # Process any exception raised by __exit__(), or it could be lost
            print(f'Exception in SSHConnCustom.__exit__(): {e}')
        return False  # if __exit__() was called due to an exception - re-raise it

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f'Exception in SSHConnCustom.__del__(): {e}')
        

if __name__ == '__main__':
    
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
    
    print('\nTesting a direct connection (explicitly create and close):')
    conn = SSHConnCustom([ssh_par_lethe])
    result = conn.run('uname -a', hide=True)
    print(result.stdout.strip())
    conn.close()
    
    print('\nTesting a proxy connection (explicitly create and close):')
    conn = SSHConnCustom([ssh_par_lethe, ssh_par_grid])
    result = conn.run('uname -a', hide=True)
    print(result.stdout.strip())
    conn.close()
    
    print('\nTesting a direct connection (using "with" clause):')
    with SSHConnCustom([ssh_par_lethe]) as conn:
        result = conn.run('uname -a', hide=True)
        print(result.stdout.strip())
        conn.close()
    
    print('\nTesting a proxy connection (using "with" clause):')
    with SSHConnCustom([ssh_par_lethe, ssh_par_grid]) as conn:
        result = conn.run('uname -a', hide=True)
        print(result.stdout.strip())
        conn.close()

