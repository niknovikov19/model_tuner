from typing import List


from ssh_conn_custom import SSHConnCustom
from ssh_fs_custom import SSHFSCustom
from ssh_params import SSHParams


class SSHClientCloseError(Exception):
    """Raised when one or more errors occur while closing SSHClient resources."""
    def __init__(self, message: str, inner_exceptions: list[Exception]):
        super().__init__(message)
        self.inner_exceptions = inner_exceptions


class SSHClient:
    """Two ssh objects: filesystem and connection for cmd line commands. """
    
    def __init__(
            self,
            ssh_par_fs: SSHParams,
            ssh_par_conn: SSHParams | List[SSHParams],
            ):
        self._ssh_par_fs = ssh_par_fs
        self.fs = SSHFSCustom(ssh_par_fs)
        self._ssh_par_conn = ssh_par_conn
        self.conn = SSHConnCustom(ssh_par_conn)
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
        
        # Close the fabric connection(s)
        try:
            self.conn.close()
        except Exception as e:
            exceptions.append(e)
        
        self._is_open = False
        
        if exceptions:
            raise SSHClientCloseError(
                'One or more errors occurred while closing SSHClient resources',
                exceptions
            )

    def __enter__(self):
        """Enter the context manager, returning self. """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exiting the 'with' block, close resources. """
        try:
            self.close()
        except Exception as e:
            print(f'Exception in SSHClient.__exit__(): {e}')
        return False
        