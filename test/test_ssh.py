import os

from fabric import Connection as FabricConnection
from fs.sshfs import SSHFS
import paramiko


def path_join(*parts, sep='/'):
    return sep.join(str(part).strip(sep) for part in parts)

# SSH parameters
hostname = 'lethe.downstate.edu'
username = 'niknovikov19'
port = 1415
private_key_path = r'C:\Users\aleks\.ssh\id_rsa_lethe'

# HPC folder
remote_dir = 'ddn/niknovikov19'

# Test ssh-based file system
private_key = paramiko.RSAKey(filename=private_key_path)
with SSHFS(
        host=hostname,
        user=username,
        port=port,
        pkey=private_key
        ) as fs:
    print(f'Contents of {remote_dir} via SSHFS')
    print(fs.listdir(remote_dir))

# Test ssh-based command running
with FabricConnection(
        host=hostname,
        user=username,
        port=port,
        connect_kwargs={'key_filename': private_key_path}
        ) as conn:
    result = conn.run('ls -l', hide=True)
    print(f'\nContents of {remote_dir} via "ls - l"')
    print(result.stdout.strip())
    
